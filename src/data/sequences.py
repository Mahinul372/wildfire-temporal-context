"""Sequence construction: from daily GeoTIFFs to prediction samples.

A sample is one (event, target date) pair:
  * inputs: the 5 consecutive days ending on day t  (t-4 .. t)
  * target: the binarised active-fire map of day t+1

Eligibility is defined ONCE, by the 5-day history requirement, and is
shared by all temporal configurations E1-E4 -- so every configuration
is evaluated on the identical sample population and only the sliced
input window differs. A sample is eligible only if the 5 input days
and the target day form 6 consecutive calendar days (no gaps), which
also guarantees no future information enters the inputs.

Splits are by fire event (never by pixel or by day); the split of each
event comes from the config, so no event appears in two partitions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

HISTORY_DAYS = 5  # E4 window; eligibility is defined by this maximum

DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})\.tif$")


@dataclass(frozen=True)
class Sample:
    event: str  # e.g. "2018/fire_21889943"
    split: str  # train / validation / test
    target_date: date
    input_dates: tuple[date, ...]  # 5 consecutive days, oldest first
    input_paths: tuple[Path, ...]
    target_path: Path

    @property
    def last_input_date(self) -> date:
        return self.input_dates[-1]


def inventory_events(raw_dir: Path) -> dict[str, dict[date, Path]]:
    """Map each event to ``{date: tif_path}`` for its downloaded days."""
    events: dict[str, dict[date, Path]] = {}
    for tif in sorted(raw_dir.rglob("*.tif")):
        m = DATE_RE.match(tif.name)
        if not m:
            continue
        d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        event = f"{tif.parent.parent.name}/{tif.parent.name}"
        events.setdefault(event, {})[d] = tif
    return events


def build_samples(
    events: dict[str, dict[date, Path]],
    split_of_event: dict[str, str],
    history_days: int = HISTORY_DAYS,
) -> list[Sample]:
    """Build all eligible samples from the inventoried events.

    A target date ``d`` is eligible iff days ``d-history_days .. d``
    are all present as consecutive calendar days.
    """
    samples: list[Sample] = []
    for event, by_date in sorted(events.items()):
        split = split_of_event[event]
        dates = sorted(by_date)
        date_set = set(dates)
        for target in dates:
            first = target - timedelta(days=history_days)
            window = [first + timedelta(days=i) for i in range(history_days + 1)]
            if all(d in date_set for d in window):
                inputs = tuple(window[:-1])
                samples.append(
                    Sample(
                        event=event,
                        split=split,
                        target_date=target,
                        input_dates=inputs,
                        input_paths=tuple(by_date[d] for d in inputs),
                        target_path=by_date[target],
                    )
                )
    return samples


def split_counts(samples: list[Sample]) -> dict[str, dict[str, int]]:
    """Count samples and distinct events per split."""
    out: dict[str, dict[str, int]] = {}
    for s in samples:
        d = out.setdefault(s.split, {"samples": 0, "events": set()})
        d["samples"] += 1
        d["events"].add(s.event)
    return {
        k: {"samples": v["samples"], "events": len(v["events"])}
        for k, v in sorted(out.items())
    }
