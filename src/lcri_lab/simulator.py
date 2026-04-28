from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd


REGIMES = ("thick", "thin", "stressed", "replenishing")


@dataclass(frozen=True)
class SimulationConfig:
    rows: int = 20_000
    levels: int = 5
    seed: int = 7
    tick_size: float = 0.01
    initial_mid: float = 100.0

    def __post_init__(self) -> None:
        if self.rows < 1:
            raise ValueError("rows must be at least 1")
        if self.levels < 1:
            raise ValueError("levels must be at least 1")
        if not math.isfinite(self.tick_size) or self.tick_size <= 0.0:
            raise ValueError("tick_size must be a finite positive value")
        if not math.isfinite(self.initial_mid) or self.initial_mid <= 0.0:
            raise ValueError("initial_mid must be a finite positive value")


def simulate_order_books(config: SimulationConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    regimes = rng.choice(REGIMES, size=config.rows, p=[0.38, 0.24, 0.18, 0.20])

    rows: list[dict[str, Union[float, int, str]]] = []
    mid = config.initial_mid

    for ts, regime in enumerate(regimes):
        params = _regime_params(regime)
        spread_ticks = int(rng.choice(params["spread_ticks"], p=params["spread_probs"]))
        spread = spread_ticks * config.tick_size
        volatility = float(rng.lognormal(params["vol_mu"], params["vol_sigma"]))
        replenishment = float(np.clip(rng.normal(params["replenishment_mean"], 0.12), 0.02, 1.0))
        latent_pressure = float(rng.normal(0.0, params["pressure_sigma"]))
        structural_bias = float(
            params["structural_bid_bias"]
            + 0.20 * (replenishment - 0.50)
            - 0.025 * (spread_ticks - 2)
            + rng.normal(0.0, 0.04)
        )

        base_depth = float(rng.lognormal(params["depth_mu"], params["depth_sigma"]))
        bid_depths, ask_depths = _depth_ladder(
            rng=rng,
            levels=config.levels,
            base_depth=base_depth,
            latent_pressure=latent_pressure,
            structural_bias=structural_bias,
            replenishment=replenishment,
            stress_multiplier=params["stress_multiplier"],
        )

        raw_imbalance = (bid_depths.sum() - ask_depths.sum()) / (bid_depths.sum() + ask_depths.sum())
        move_prob = _move_probability(
            raw_imbalance=raw_imbalance,
            latent_pressure=latent_pressure,
            spread=spread,
            volatility=volatility,
            replenishment=replenishment,
            regime=regime,
        )
        direction = 1 if rng.random() < move_prob else -1
        move_ticks = rng.poisson(params["move_lambda"]) + (1 if rng.random() < volatility else 0)
        mid_move = direction * move_ticks * config.tick_size
        next_mid = mid + mid_move

        row: dict[str, Union[float, int, str]] = {
            "timestamp": ts,
            "regime": regime,
            "mid": mid,
            "next_mid": next_mid,
            "spread": spread,
            "spread_ticks": spread_ticks,
            "volatility": volatility,
            "replenishment_rate": replenishment,
            "latent_pressure": latent_pressure,
            "structural_bias": structural_bias,
            "future_direction": 1 if next_mid > mid else 0,
            "future_return_ticks": mid_move / config.tick_size,
        }

        for level in range(config.levels):
            row[f"bid_px_{level + 1}"] = mid - spread / 2 - level * config.tick_size
            row[f"ask_px_{level + 1}"] = mid + spread / 2 + level * config.tick_size
            row[f"bid_sz_{level + 1}"] = float(bid_depths[level])
            row[f"ask_sz_{level + 1}"] = float(ask_depths[level])

        rows.append(row)
        mid = next_mid

    return pd.DataFrame(rows)


def _regime_params(regime: str) -> dict[str, object]:
    params = {
        "thick": {
            "depth_mu": 7.2,
            "depth_sigma": 0.25,
            "spread_ticks": [1, 2, 3],
            "spread_probs": [0.75, 0.20, 0.05],
            "vol_mu": -2.6,
            "vol_sigma": 0.25,
            "replenishment_mean": 0.78,
            "pressure_sigma": 0.45,
            "stress_multiplier": 0.95,
            "structural_bid_bias": 0.08,
            "move_lambda": 0.25,
        },
        "thin": {
            "depth_mu": 5.7,
            "depth_sigma": 0.45,
            "spread_ticks": [1, 2, 3, 4],
            "spread_probs": [0.20, 0.42, 0.28, 0.10],
            "vol_mu": -2.0,
            "vol_sigma": 0.35,
            "replenishment_mean": 0.38,
            "pressure_sigma": 0.70,
            "stress_multiplier": 1.10,
            "structural_bid_bias": -0.04,
            "move_lambda": 0.45,
        },
        "stressed": {
            "depth_mu": 5.35,
            "depth_sigma": 0.65,
            "spread_ticks": [2, 3, 4, 5, 6],
            "spread_probs": [0.12, 0.24, 0.30, 0.22, 0.12],
            "vol_mu": -1.45,
            "vol_sigma": 0.45,
            "replenishment_mean": 0.22,
            "pressure_sigma": 1.10,
            "stress_multiplier": 1.40,
            "structural_bid_bias": -0.12,
            "move_lambda": 0.80,
        },
        "replenishing": {
            "depth_mu": 6.7,
            "depth_sigma": 0.35,
            "spread_ticks": [1, 2, 3],
            "spread_probs": [0.50, 0.36, 0.14],
            "vol_mu": -2.2,
            "vol_sigma": 0.30,
            "replenishment_mean": 0.90,
            "pressure_sigma": 0.55,
            "stress_multiplier": 0.80,
            "structural_bid_bias": 0.14,
            "move_lambda": 0.32,
        },
    }
    return params[regime]


def _depth_ladder(
    rng: np.random.Generator,
    levels: int,
    base_depth: float,
    latent_pressure: float,
    structural_bias: float,
    replenishment: float,
    stress_multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    decay = np.exp(-0.18 * np.arange(levels))
    noise_bid = rng.lognormal(0.0, 0.20 * stress_multiplier, size=levels)
    noise_ask = rng.lognormal(0.0, 0.20 * stress_multiplier, size=levels)
    pressure = structural_bias + np.tanh(latent_pressure) * (0.22 + 0.20 * (1.0 - replenishment))
    pressure = float(np.clip(pressure, -0.75, 0.75))

    bid_depths = base_depth * decay * noise_bid * (1.0 + pressure)
    ask_depths = base_depth * decay * noise_ask * (1.0 - pressure)
    return np.maximum(bid_depths, 1.0), np.maximum(ask_depths, 1.0)


def _move_probability(
    raw_imbalance: float,
    latent_pressure: float,
    spread: float,
    volatility: float,
    replenishment: float,
    regime: str,
) -> float:
    regime_gain = {
        "thick": 1.10,
        "thin": 1.45,
        "stressed": 1.75,
        "replenishing": 0.85,
    }[regime]
    liquidity_penalty = 0.35 * spread + 0.50 * replenishment
    score = regime_gain * (0.45 * raw_imbalance + 0.85 * latent_pressure)
    score += 0.45 * volatility - liquidity_penalty
    return float(1.0 / (1.0 + np.exp(-score)))
