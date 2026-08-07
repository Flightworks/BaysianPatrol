"""Canonical Gymnasium environment for BaysianPatrol v2.3.1.

The policy selects a relative search waypoint. A deterministic autopilot flies to
that waypoint and a non-learned safety shield guarantees a conservative return
to the frigate before Bingo fuel.
"""
from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.ndimage import gaussian_filter, shift

try:
    from .navigation import advance_towards_waypoint, derive_initial_target_truth, estimate_travel_minutes
except ImportError:
    from navigation import advance_towards_waypoint, derive_initial_target_truth, estimate_travel_minutes


class BaysianPatrolEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        reward_detection: float = 10.0,
        reward_probability: float = 2.0,
        penalty_time: float = 0.01,
        penalty_bingo: float = 10.0,
        curriculum_level: int = 4,
        reward_exploration: float | None = None,
    ):
        super().__init__()
        self.grid_size = 32
        self.area_width = self.area_height = 100.0
        self.half_width = self.area_width / 2.0
        self.half_height = self.area_height / 2.0
        self.min_x = self.min_y = -50.0
        self.max_x = self.max_y = 50.0
        self.dt = 1.0
        self.max_steps = 180
        self.max_fuel = 180.0
        self.max_speed = 140.0
        self.min_speed = 60.0
        self.radar_range = 15.0
        self.bingo_buffer = 15.0
        self.rtb_radius = 1.0
        self.curriculum_level = int(curriculum_level)

        self.reward_detection = float(reward_detection)
        if reward_exploration is not None:
            reward_probability = reward_exploration
        self.reward_probability = float(reward_probability)
        self.penalty_time = float(penalty_time)
        self.penalty_bingo = float(penalty_bingo)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0.0, 1.0, shape=(2, 32, 32), dtype=np.float32),
            "vector": spaces.Box(-1.0, 1.0, shape=(10,), dtype=np.float32),
        })

        cell = self.area_width / self.grid_size
        self.grid_x = np.linspace(self.min_x + cell / 2, self.max_x - cell / 2, self.grid_size)
        self.grid_y = np.linspace(self.min_y + cell / 2, self.max_y - cell / 2, self.grid_size)
        self.mesh_x, self.mesh_y = np.meshgrid(self.grid_x, self.grid_y)

    @staticmethod
    def _normalise_probability(values: np.ndarray) -> np.ndarray:
        values = np.maximum(values, 1e-12)
        return (values / values.sum()).astype(np.float32)

    def _initial_belief(self) -> np.ndarray:
        sigma = 10.0 if self.curriculum_level <= 1 else 15.0
        p = np.exp(-0.5 * (((self.mesh_x-self.datum_x)/sigma)**2 + ((self.mesh_y-self.datum_y)/sigma)**2))
        return self._normalise_probability(p)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        self.steps = 0
        self.elapsed_min = 0.0
        self.frigate_x = float(self.np_random.uniform(-35.0, 35.0))
        self.frigate_y = float(self.np_random.uniform(-35.0, 35.0))
        self.datum_x = float(self.np_random.uniform(-20.0, 20.0))
        self.datum_y = float(self.np_random.uniform(-20.0, 20.0))
        self.helico_x, self.helico_y = self.frigate_x, self.frigate_y
        self.helico_heading = float(self.np_random.uniform(-math.pi, math.pi))
        self.current_speed = self.max_speed
        self.fuel_remaining = self.max_fuel
        self.rtb_active = False
        self.wind_speed = float(max(0.0, self.np_random.normal(15.0, 4.0)))
        self.wind_from_direction = float(self.np_random.normal(270.0, 15.0) % 360.0)

        if self.curriculum_level <= 1:
            self.target_speed = 0.0
            self.target_heading = math.radians(45.0)
        else:
            self.target_speed = float(np.clip(self.np_random.normal(20.0, 3.0), 5.0, 35.0))
            self.target_heading = math.radians(float(self.np_random.normal(45.0, 12.0)))
        spatial_offset_x = float(self.np_random.normal(0.0, 2.0))
        spatial_offset_y = float(self.np_random.normal(0.0, 2.0))
        datum_time_offset_minutes = float(self.np_random.normal(0.0, 10.0))
        self.target_current_speed = self.wind_speed * 0.025
        self.target_current_heading = (self.wind_from_direction + 180.0 + 15.0) % 360.0
        initial_truth = derive_initial_target_truth(
            datum_x=self.datum_x,
            datum_y=self.datum_y,
            spatial_offset_x=spatial_offset_x,
            spatial_offset_y=spatial_offset_y,
            time_offset_minutes=datum_time_offset_minutes,
            speed=self.target_speed,
            heading=math.degrees(self.target_heading),
            current_speed=self.target_current_speed,
            current_heading=self.target_current_heading,
        )
        self.target_x = initial_truth["x"]
        self.target_y = initial_truth["y"]
        self.belief_speed = 0.0 if self.curriculum_level <= 1 else 20.0
        self.belief_heading = math.radians(45.0)
        self.grid_p = self._initial_belief()
        self.scan_memory = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        return self._get_obs(), self._info("flying")

    def _peak_position(self) -> tuple[float, float]:
        idx = int(np.argmax(self.grid_p))
        j, i = divmod(idx, self.grid_size)
        return float(self.grid_x[i]), float(self.grid_y[j])

    def _entropy(self) -> float:
        p = np.maximum(self.grid_p.astype(np.float64), 1e-12)
        return float(np.clip(-(p*np.log(p)).sum() / math.log(p.size), 0.0, 1.0))

    def _return_time(self) -> float:
        return estimate_travel_minutes(
            x=self.helico_x,
            y=self.helico_y,
            waypoint_x=self.frigate_x,
            waypoint_y=self.frigate_y,
            airspeed=self.max_speed,
            wind_speed=self.wind_speed,
            wind_from_direction=self.wind_from_direction,
        )

    def _fuel_margin(self) -> float:
        return self.fuel_remaining - self._return_time() - self.bingo_buffer

    def _get_obs(self) -> dict[str, np.ndarray]:
        peak_x, peak_y = self._peak_position()
        peak = float(self.grid_p.max())
        belief_obs = self.grid_p / peak if peak > 0 else self.grid_p
        vector = np.array([
            math.sin(self.helico_heading),
            math.cos(self.helico_heading),
            self.current_speed / self.max_speed,
            np.clip(self._fuel_margin() / self.max_fuel, -1.0, 1.0),
            np.clip((self.frigate_x-self.helico_x) / self.half_width, -1.0, 1.0),
            np.clip((self.frigate_y-self.helico_y) / self.half_height, -1.0, 1.0),
            np.clip((peak_x-self.helico_x) / self.half_width, -1.0, 1.0),
            np.clip((peak_y-self.helico_y) / self.half_height, -1.0, 1.0),
            self._entropy(),
            np.clip(self.elapsed_min / self.max_fuel, 0.0, 1.0),
        ], dtype=np.float32)
        return {"grid": np.stack((belief_obs, self.scan_memory)).astype(np.float32), "vector": vector}

    def _predict_belief(self) -> None:
        cell_nm = self.area_width / self.grid_size
        dx_cells = self.belief_speed * math.sin(self.belief_heading) / 60.0 * self.dt / cell_nm
        dy_cells = self.belief_speed * math.cos(self.belief_heading) / 60.0 * self.dt / cell_nm
        predicted = shift(self.grid_p, shift=(dy_cells, dx_cells), order=1, mode="nearest", prefilter=False)
        sigma_cells = (0.10 if self.curriculum_level <= 2 else 0.18) * math.sqrt(self.dt)
        self.grid_p = self._normalise_probability(gaussian_filter(predicted, sigma=sigma_cells, mode="nearest"))

    def _move_target(self) -> None:
        if self.curriculum_level >= 3:
            self.target_heading += float(self.np_random.normal(0.0, math.radians(1.5)))
        distance = self.target_speed / 60.0 * self.dt
        current_distance = self.target_current_speed / 60.0 * self.dt
        current_heading = math.radians(self.target_current_heading)
        self.target_x += distance * math.sin(self.target_heading) + current_distance * math.sin(current_heading)
        self.target_y += distance * math.cos(self.target_heading) + current_distance * math.cos(current_heading)
        if self.target_x < self.min_x or self.target_x > self.max_x:
            self.target_x = float(np.clip(self.target_x, self.min_x, self.max_x))
            self.target_heading = -self.target_heading
        if self.target_y < self.min_y or self.target_y > self.max_y:
            self.target_y = float(np.clip(self.target_y, self.min_y, self.max_y))
            self.target_heading = math.pi - self.target_heading

    def _fly_towards(self, waypoint_x: float, waypoint_y: float) -> None:
        waypoint_x = float(np.clip(waypoint_x, self.min_x, self.max_x))
        waypoint_y = float(np.clip(waypoint_y, self.min_y, self.max_y))
        next_state = advance_towards_waypoint(
            x=self.helico_x,
            y=self.helico_y,
            waypoint_x=waypoint_x,
            waypoint_y=waypoint_y,
            airspeed=self.max_speed,
            dt_minutes=self.dt,
            wind_speed=self.wind_speed,
            wind_from_direction=self.wind_from_direction,
        )
        self.helico_x = next_state["x"]
        self.helico_y = next_state["y"]
        self.helico_heading = math.radians(next_state["air_heading"])
        self.current_speed = self.max_speed

    def _negative_scan_update(self) -> float:
        dist = np.hypot(self.mesh_x-self.helico_x, self.mesh_y-self.helico_y)
        pdet = np.clip(1.0 - dist/self.radar_range, 0.0, 0.9)
        before = float((self.grid_p*pdet).sum())
        self.grid_p = self._normalise_probability(self.grid_p * (1.0-pdet))
        decay = math.exp(-math.log(2.0)*self.dt/20.0)
        self.scan_memory *= decay
        self.scan_memory[pdet > 0.05] = np.maximum(self.scan_memory[pdet > 0.05], pdet[pdet > 0.05]).astype(np.float32)
        return before

    def expert_action(self) -> np.ndarray:
        peak_x, peak_y = self._peak_position()
        return np.array([
            np.clip((peak_x-self.helico_x)/self.half_width, -1.0, 1.0),
            np.clip((peak_y-self.helico_y)/self.half_height, -1.0, 1.0),
        ], dtype=np.float32)

    def _info(self, outcome: str, *, information_gain: float = 0.0, out_of_bounds: bool = False, bingo: bool = False) -> dict[str, Any]:
        return {
            "outcome": outcome,
            "intercepted": outcome == "intercepted",
            "safe_rtb": outcome == "safe_rtb",
            "bingo_fail": bool(bingo),
            "out_of_bounds": bool(out_of_bounds),
            "information_gain": float(information_gain),
            "fuel_margin": float(self._fuel_margin()),
            "fuel_remaining": float(self.fuel_remaining),
            "elapsed_min": float(self.elapsed_min),
        }

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.steps += 1
        self.elapsed_min += self.dt
        self._move_target()
        self._predict_belief()

        if self._fuel_margin() <= self.dt:
            self.rtb_active = True

        if self.rtb_active:
            waypoint_x, waypoint_y = self.frigate_x, self.frigate_y
        else:
            waypoint_x = self.helico_x + float(action[0]) * self.half_width
            waypoint_y = self.helico_y + float(action[1]) * self.half_height
        self._fly_towards(waypoint_x, waypoint_y)
        self.fuel_remaining = max(0.0, self.fuel_remaining-self.dt)

        dist_target = math.hypot(self.target_x-self.helico_x, self.target_y-self.helico_y)
        pdet_target = float(np.clip(1.0-dist_target/self.radar_range, 0.0, 0.95))
        detected = dist_target < 0.8 or (pdet_target > 0.05 and self.np_random.random() < pdet_target)

        if detected:
            return self._get_obs(), float(self.reward_detection), True, False, self._info("intercepted")

        information_gain = self._negative_scan_update()
        reward = self.reward_probability*information_gain - self.penalty_time*self.dt

        dist_frigate = math.hypot(self.helico_x-self.frigate_x, self.helico_y-self.frigate_y)
        if self.rtb_active and dist_frigate <= self.rtb_radius:
            return self._get_obs(), float(reward), True, False, self._info("safe_rtb", information_gain=information_gain)

        out = not (self.min_x <= self.helico_x <= self.max_x and self.min_y <= self.helico_y <= self.max_y)
        if out:
            return self._get_obs(), float(reward-self.penalty_bingo), True, False, self._info("out_of_bounds", information_gain=information_gain, out_of_bounds=True)

        bingo = self.fuel_remaining < self._return_time()
        if bingo:
            return self._get_obs(), float(reward-self.penalty_bingo), True, False, self._info("bingo", information_gain=information_gain, bingo=True)

        truncated = self.steps >= self.max_steps or self.fuel_remaining <= 0.0
        outcome = "time_limit" if truncated else "flying"
        return self._get_obs(), float(reward), False, bool(truncated), self._info(outcome, information_gain=information_gain)
