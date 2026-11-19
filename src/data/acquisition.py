"""Acquisition of a small subset of WildfireSpreadTS without downloading
the full 48.4 GB archive.

Method (verified against the live Zenodo record 8006177):
  1. The archive is a single ZIP (WildfireSpreadTS.zip, 48,359,369,821
     bytes) served by Zenodo with HTTP Range support (206 Partial
     Content, ``Accept-Ranges: bytes``).
  2. The ZIP central directory (ZIP64) is fetched with one Range request
     and parsed locally, giving every member's name, compressed size,
     uncompressed size and local-header offset -- without downloading
     any file data.
  3. Individual GeoTIFF members are extracted with ``remotezip``,
     which issues Range requests for exactly the bytes of that member.

Nothing here downloads the whole archive. Downloaded data goes to
``data/raw/`` and is excluded from git (see .gitignore).
"""

from __future__ import annotations

import json
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from remotezip import RemoteZip

ZIP_URL = (
    "https://zenodo.org/api/records/8006177/files/WildfireSpreadTS.zip/content"
)
ZIP_SIZE = 48_359_369_821


@dataclass(frozen=True)
class ZipMember:
    """One entry of the ZIP central directory."""

    name: str
    method: int  # 0 = stored, 8 = deflate
    compress_size: int
    file_size: int
    header_offset: int

    @property
    def is_tif(self) -> bool:
        return self.name.endswith(".tif")


def fetch_central_directory(
    url: str = ZIP_URL, size: int = ZIP_SIZE, timeout: int = 120
) -> list[ZipMember]:
    """Fetch and parse the ZIP64 central directory via Range requests.

    Returns the full member list (14,218 entries for WildfireSpreadTS).
    Raises on HTTP errors or an unparsable directory.
    """
    # 1. Tail of the file: ZIP64 end-of-central-directory locator + EOCD.
    tail_len = 8 * 1024 * 1024
    r = requests.get(
        url,
        headers={"Range": f"bytes={size - tail_len}-{size - 1}"},
        timeout=timeout,
    )
    r.raise_for_status()
    tail = r.content
    locator = tail.rfind(b"PK\x06\x07")
    if locator < 0:
        raise ValueError("ZIP64 EOCD locator not found in archive tail")
    zip64_eocd_offset = struct.unpack("<Q", tail[locator + 8 : locator + 16])[0]

    # 2. The 56-byte ZIP64 EOCD record.
    r = requests.get(
        url,
        headers={"Range": f"bytes={zip64_eocd_offset}-{zip64_eocd_offset + 55}"},
        timeout=timeout,
    )
    r.raise_for_status()
    rec = r.content
    sig, _sz, _vmade, _vneed, _disk, _cddisk, _ent_disk, ent_total, cd_size, cd_off = (
        struct.unpack("<IQHHIIQQQQ", rec[:56])
    )
    if sig != 0x06064B50:
        raise ValueError(f"bad ZIP64 EOCD signature: {sig:#x}")

    # 3. The central directory itself.
    r = requests.get(
        url,
        headers={"Range": f"bytes={cd_off}-{cd_off + cd_size - 1}"},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.content
    members: list[ZipMember] = []
    pos = 0
    while pos < len(data):
        if data[pos : pos + 4] != b"PK\x01\x02":
            raise ValueError(f"central directory parse failed at offset {pos}")
        fields = struct.unpack("<HHHHHHIIIHHHHHII", data[pos + 4 : pos + 46])
        (_vmade, _vneed, _flags, method, _mtime, _mdate, _crc,
         csize, usize, nlen, elen, clen, _dstart, _iattr, _eattr,
         lho) = fields
        name = data[pos + 46 : pos + 46 + nlen].decode("utf-8", "replace")
        members.append(ZipMember(name, method, csize, usize, lho))
        pos += 46 + nlen + elen + clen
    if len(members) != ent_total:
        raise ValueError(
            f"parsed {len(members)} members but EOCD says {ent_total}"
        )
    return members


def list_fire_events(members: Iterable[ZipMember]) -> dict[str, list[str]]:
    """Group GeoTIFF members by fire event (``<year>/fire_<id>``).

    Returns ``{event_path: [member_name, ...]}`` sorted by date.
    """
    events: dict[str, list[str]] = {}
    for m in members:
        if not m.is_tif:
            continue
        parts = m.name.split("/")
        if len(parts) != 3:
            continue
        event = f"{parts[0]}/{parts[1]}"
        events.setdefault(event, []).append(m.name)
    for names in events.values():
        names.sort()
    return events


def _download_one(
    url: str, member: ZipMember, out_path: Path, retries: int = 3
) -> dict:
    """Extract one member with remotezip. Skips files already present
    with the expected uncompressed size (resumable)."""
    if out_path.exists() and out_path.stat().st_size == member.file_size:
        return {"name": member.name, "status": "skipped", "bytes": member.file_size}
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with RemoteZip(url) as rz:
                data = rz.read(member.name)
            if len(data) != member.file_size:
                raise ValueError(
                    f"size mismatch for {member.name}: got {len(data)}, "
                    f"expected {member.file_size}"
                )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(out_path.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(out_path)
            return {"name": member.name, "status": "downloaded",
                    "bytes": member.file_size}
        except Exception as exc:  # noqa: BLE001 -- retry then report
            last_exc = exc
            time.sleep(2 * attempt)
    return {"name": member.name, "status": "failed",
            "error": repr(last_exc), "bytes": 0}


def download_events(
    events: dict[str, list[str]],
    members: dict[str, ZipMember],
    out_dir: Path,
    workers: int = 4,
    url: str = ZIP_URL,
) -> list[dict]:
    """Download every GeoTIFF of the given events into ``out_dir``.

    ``events`` maps ``<year>/fire_<id>`` to member names. Files are
    written to ``out_dir/<year>/fire_<id>/<date>.tif``. Returns a
    per-file result list (downloaded / skipped / failed).
    """
    jobs: list[tuple[ZipMember, Path]] = []
    for event, names in events.items():
        for name in names:
            member = members[name]
            out_path = out_dir / name
            jobs.append((member, out_path))
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_one, url, m, p): m.name for m, p in jobs
        }
        for fut in as_completed(futures):
            results.append(fut.result())
    return results


def write_manifest(results: list[dict], path: Path) -> None:
    """Write the download manifest (per-file status + totals)."""
    total = sum(r["bytes"] for r in results if r["status"] != "failed")
    manifest = {
        "source": ZIP_URL,
        "files": results,
        "n_downloaded": sum(1 for r in results if r["status"] == "downloaded"),
        "n_skipped": sum(1 for r in results if r["status"] == "skipped"),
        "n_failed": sum(1 for r in results if r["status"] == "failed"),
        "total_bytes": total,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
