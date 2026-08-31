"""Synthetic cell-tower congestion data — the concrete telecom demo built
on top of the existing distributed-training machinery.

**This is illustrative, synthetic data. No real carrier, subscriber, or
network measurement of any kind went into it.** Every "tower" is a
hand-built archetype (a smooth expected-load curve over hour-of-day),
sampled with noise. Nothing here is fitted to, or claims to resemble,
any real operator's actual traffic.

The story: N cell towers, each seeing its own local pattern of traffic
(a business district peaks 9-to-5 on weekdays; a residential area peaks
evenings; an entertainment district peaks weekend nights; a transit hub
has two rush-hour spikes) — genuinely non-IID, the honest hard case this
project has used throughout (see tron/training/data.py). Each tower
predicts its own next-interval congestion tier (low/medium/high/critical)
from recent signals. Training happens the same way every other
distributed run in this repo does: local steps on each tower's own data,
periodic sync of a small weight delta through the master — raw traffic
data never leaves the tower. The point of this module is only the
dataset; the training mechanism (local SGD, the real wire transport, the
Grid, WAN validation) is everything already built and tested elsewhere in
this repo, unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

NUM_FEATURES = 6  # hour_sin, hour_cos, is_weekend, recent_avg_load, recent_load_trend, active_devices_norm
NUM_TIERS = 4      # 0=low, 1=medium, 2=high, 3=critical
TIER_NAMES = ["low", "medium", "high", "critical"]
_TIER_EDGES = (0.35, 0.60, 0.85)  # load thresholds separating the 4 tiers


@dataclass(frozen=True)
class TowerProfile:
    """A named archetype: `expected_load(hour, is_weekend)` returns a
    value in roughly [0, 1] — the "true" congestion level a tower of this
    kind tends toward at that hour, before noise. Built from Gaussian
    bumps over hour-of-day so the curve is smooth and has an honest
    interpretation, not an arbitrary lookup table."""

    name: str
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray]


def _bump(hour: np.ndarray, center: float, width: float) -> np.ndarray:
    """A Gaussian bump over the 24h clock, wrapping around midnight."""
    d = np.minimum(np.abs(hour - center), 24.0 - np.abs(hour - center))
    return np.exp(-0.5 * (d / width) ** 2)


def _business_district(hour: np.ndarray, is_weekend: np.ndarray) -> np.ndarray:
    weekday_curve = 0.15 + 0.85 * _bump(hour, center=13.0, width=4.5)
    weekend_curve = 0.05 + 0.20 * _bump(hour, center=13.0, width=4.5)
    return np.where(is_weekend, weekend_curve, weekday_curve)


def _residential(hour: np.ndarray, is_weekend: np.ndarray) -> np.ndarray:
    base = 0.10 + 0.75 * _bump(hour, center=20.0, width=3.0) + 0.15 * _bump(hour, center=8.0, width=1.5)
    return base + np.where(is_weekend, 0.10, 0.0)  # slightly busier all day on weekends


def _entertainment_district(hour: np.ndarray, is_weekend: np.ndarray) -> np.ndarray:
    weekday_curve = 0.08 + 0.55 * _bump(hour, center=22.0, width=2.5)
    weekend_curve = 0.10 + 0.90 * _bump(hour, center=23.5, width=3.0)
    return np.where(is_weekend, weekend_curve, weekday_curve)


def _transit_hub(hour: np.ndarray, is_weekend: np.ndarray) -> np.ndarray:
    weekday_curve = 0.10 + 0.70 * _bump(hour, center=8.0, width=1.2) + 0.70 * _bump(hour, center=17.5, width=1.4)
    weekend_curve = 0.08 + 0.30 * _bump(hour, center=13.0, width=5.0)
    return np.where(is_weekend, weekend_curve, weekday_curve)


TOWER_PROFILES = [
    TowerProfile("business-district", _business_district),
    TowerProfile("residential", _residential),
    TowerProfile("entertainment-district", _entertainment_district),
    TowerProfile("transit-hub", _transit_hub),
]


def _load_to_tier(load: np.ndarray) -> np.ndarray:
    return np.digitize(load, _TIER_EDGES)  # 0..3


def _make_tower_samples(profile: TowerProfile, num_samples: int, rng: np.random.Generator):
    """One tower's synthetic hourly readings. Returns (x, y): x is
    NUM_FEATURES columns, y is the congestion tier 0..3."""
    hour = rng.uniform(0.0, 24.0, size=num_samples)
    is_weekend = rng.integers(0, 2, size=num_samples).astype(bool)

    true_load = profile.fn(hour, is_weekend)
    true_load = np.clip(true_load + rng.normal(scale=0.04, size=num_samples), 0.0, 1.0)
    tier = _load_to_tier(true_load)

    # Features a real monitoring feed would plausibly hand you — noisy
    # observations correlated with, but not identical to, the ground
    # truth the label is drawn from (a model earning its accuracy has to
    # actually learn the hour/weekend pattern, not just copy a leaked
    # feature).
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    recent_avg_load = np.clip(true_load + rng.normal(scale=0.10, size=num_samples), 0.0, 1.0)
    recent_load_trend = np.clip(rng.normal(scale=0.15, size=num_samples), -1.0, 1.0)
    active_devices_norm = np.clip(0.7 * true_load + 0.3 * rng.uniform(size=num_samples), 0.0, 1.0)

    x = np.stack(
        [hour_sin, hour_cos, is_weekend.astype(float), recent_avg_load, recent_load_trend, active_devices_norm],
        axis=1,
    )
    return x.astype(np.float64), tier.astype(np.int64)


def build_tower_problem(
    *, num_shards: int, dataset_config: dict
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, int, int]:
    """The `dataset_factory` hook `TrainingSession` calls when
    `problem="telecom_congestion"`. Returns
    (shards, x_test, y_test, num_features, num_classes) — the same shape
    `TrainingSession._build_problem`'s default (Gaussian-blob
    classification) path produces, so nothing downstream (the wire
    protocol, the spine, the Grid, outcome scoring) has to know or care
    which problem is running.

    Each of `num_shards` towers is assigned one of TOWER_PROFILES,
    cycling if there are more towers than profiles — genuinely
    non-IID by construction (a business-district tower's data looks
    nothing like a residential one's), not shuffled to look easier than
    the real hard case.
    """
    cfg = dataset_config or {}
    seed = int(cfg.get("seed", 0))
    samples_per_tower = int(cfg.get("samples_per_tower", 400))
    test_samples_per_profile = int(cfg.get("test_samples_per_profile", 200))

    shards = []
    for i in range(num_shards):
        profile = TOWER_PROFILES[i % len(TOWER_PROFILES)]
        rng = np.random.default_rng(seed * 1000 + i + 1)
        shards.append(_make_tower_samples(profile, samples_per_tower, rng))

    # Held-out set: drawn from every profile, not just the ones this run's
    # towers happen to use, so accuracy reflects the general pattern
    # learned, not one memorized per-tower quirk.
    test_rng = np.random.default_rng(seed * 1000 + 999_999)
    xs, ys = [], []
    for profile in TOWER_PROFILES:
        x, y = _make_tower_samples(profile, test_samples_per_profile, test_rng)
        xs.append(x)
        ys.append(y)
    x_test = np.concatenate(xs, axis=0)
    y_test = np.concatenate(ys, axis=0)

    return shards, x_test, y_test, NUM_FEATURES, NUM_TIERS


def tower_profile_for_shard(shard_idx: int) -> str:
    """Which archetype a given shard index was assigned — for labeling
    output/the Grid, not used by training itself."""
    return TOWER_PROFILES[shard_idx % len(TOWER_PROFILES)].name
