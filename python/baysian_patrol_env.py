import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BaysianPatrolEnv(gym.Env):
    """Maritime SAR search environment with reproducible stochastic dynamics."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        grid_dim: int = 32,
        search_area_width: float = 120.0,
        search_area_height: float = 120.0,
        dt: float = 1.0,
        max_endurance: float = 180.0,
        helico_max_speed: float = 140.0,
        radar_base_range: float = 15.0,
        max_turn_rate_deg_min: float = 180.0,
        reward_detection: float = 500.0,
        reward_exploration: float = 10.0,
        penalty_time: float = 0.1,
        penalty_bingo: float = 1000.0,
    ):
        super().__init__()
        self.grid_dim = grid_dim
        self.search_area_width = search_area_width
        self.search_area_height = search_area_height
        self.dt = dt
        self.max_endurance = max_endurance
        self.helico_max_speed = helico_max_speed
        self.radar_base_range = radar_base_range
        self.max_turn_rate_rad_step = math.radians(max_turn_rate_deg_min) * dt
        self.reward_detection = reward_detection
        self.reward_exploration = reward_exploration
        self.penalty_time = penalty_time
        self.penalty_bingo = penalty_bingo

        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0.0, 1.0, shape=(2, grid_dim, grid_dim), dtype=np.float32),
            "vector": spaces.Box(-1.0, 1.0, shape=(10,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._reset_state()

    def _rng(self):
        return getattr(self, "np_random", np.random.default_rng())

    def _reset_state(self):
        rng = self._rng()
        hw, hh = self.search_area_width / 2.0, self.search_area_height / 2.0
        self.frigate_x = rng.uniform(-0.35 * self.search_area_width, 0.35 * self.search_area_width)
        self.frigate_y = rng.uniform(-0.35 * self.search_area_height, 0.35 * self.search_area_height)
        self.datum_x = rng.uniform(-0.35 * self.search_area_width, 0.35 * self.search_area_width)
        self.datum_y = rng.uniform(-0.35 * self.search_area_height, 0.35 * self.search_area_height)
        self.target_x = self.datum_x + rng.normal(0.0, 2.0)
        self.target_y = self.datum_y + rng.normal(0.0, 2.0)
        self.target_speed = rng.uniform(8.0, 22.0)
        self.target_heading = rng.uniform(0.0, 2.0 * math.pi)
        self.helico_x, self.helico_y = self.frigate_x, self.frigate_y
        self.helico_heading = rng.uniform(0.0, 2.0 * math.pi)
        self.helico_speed = 100.0
        self.fuel_remaining = self.max_endurance
        self.t = 0.0
        self.wind_speed = rng.uniform(5.0, 25.0)
        self.wind_dir = rng.uniform(0.0, 360.0)
        self.grid_p = np.zeros((self.grid_dim, self.grid_dim), dtype=np.float32)
        self.grid_scanned = np.zeros((self.grid_dim, self.grid_dim), dtype=np.float32)
        xs = np.linspace(-hw, hw, self.grid_dim)
        ys = np.linspace(-hh, hh, self.grid_dim)
        xx, yy = np.meshgrid(xs, ys)
        dist_sq = (xx - self.datum_x) ** 2 + (yy - self.datum_y) ** 2
        self.grid_p = np.exp(-dist_sq / (2.0 * 15.0**2)).astype(np.float32)
        self.grid_p /= np.sum(self.grid_p) + 1e-8
        self.intercepted = False
        self.bingo_failed = False
        self.out_of_bounds = False
        self.target_bounced = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def _get_obs(self):
        grid_obs = np.stack([self.grid_p, self.grid_scanned], axis=0).astype(np.float32)
        x_norm = np.clip(self.helico_x / (self.search_area_width / 2.0), -1.0, 1.0)
        y_norm = np.clip(self.helico_y / (self.search_area_height / 2.0), -1.0, 1.0)
        dist_frigate = math.hypot(self.helico_x - self.frigate_x, self.helico_y - self.frigate_y)
        vector_obs = np.array([
            x_norm,
            y_norm,
            math.sin(self.helico_heading),
            math.cos(self.helico_heading),
            np.clip(self.helico_speed / self.helico_max_speed, 0.0, 1.0),
            np.clip(self.fuel_remaining / self.max_endurance, 0.0, 1.0),
            np.clip(dist_frigate / self.search_area_width, 0.0, 1.0),
            np.clip((self.wind_speed * math.sin(math.radians(self.wind_dir))) / 50.0, -1.0, 1.0),
            np.clip((self.wind_speed * math.cos(math.radians(self.wind_dir))) / 50.0, -1.0, 1.0),
            np.clip(self.radar_base_range / 30.0, 0.0, 1.0),
        ], dtype=np.float32)
        return {"grid": grid_obs, "vector": vector_obs}

    def _move_target(self):
        step = (self.target_speed / 60.0) * self.dt
        self.target_x += step * math.sin(self.target_heading)
        self.target_y += step * math.cos(self.target_heading)
        hw, hh = self.search_area_width / 2.0, self.search_area_height / 2.0
        bounced = False
        if self.target_x < -hw or self.target_x > hw:
            self.target_x = float(np.clip(self.target_x, -hw, hw))
            self.target_heading = -self.target_heading
            bounced = True
        if self.target_y < -hh or self.target_y > hh:
            self.target_y = float(np.clip(self.target_y, -hh, hh))
            self.target_heading = math.pi - self.target_heading
            bounced = True
        self.target_heading %= 2.0 * math.pi
        self.target_bounced = bounced

    def step(self, action):
        rng = self._rng()
        action = np.asarray(np.clip(action, -1.0, 1.0), dtype=np.float32)
        prev_x, prev_y = self.helico_x, self.helico_y
        self.helico_heading = (self.helico_heading + float(action[0]) * self.max_turn_rate_rad_step) % (2.0 * math.pi)
        self.helico_speed = float(np.clip(self.helico_speed + float(action[1]) * (20.0 * self.dt / 60.0), 60.0, self.helico_max_speed))
        distance = self.helico_speed / 60.0 * self.dt
        self.helico_x += distance * math.sin(self.helico_heading)
        self.helico_y += distance * math.cos(self.helico_heading)
        self._move_target()
        self.fuel_remaining -= self.dt
        self.t += self.dt

        xs = np.linspace(-self.search_area_width / 2.0, self.search_area_width / 2.0, self.grid_dim)
        ys = np.linspace(-self.search_area_height / 2.0, self.search_area_height / 2.0, self.grid_dim)
        xx, yy = np.meshgrid(xs, ys)
        drift_x = self.datum_x + self.target_speed / 60.0 * self.t * math.sin(self.target_heading)
        drift_y = self.datum_y + self.target_speed / 60.0 * self.t * math.cos(self.target_heading)
        # Uncertainty expands slowly with time instead of remaining unrealistically fixed.
        sigma = 5.0 + 0.15 * math.sqrt(max(0.0, self.t))
        self.grid_p = np.exp(-0.5 * (((xx - drift_x) / sigma) ** 2 + ((yy - drift_y) / sigma) ** 2)).astype(np.float32)
        self.grid_p /= np.sum(self.grid_p) + 1e-8

        decay = math.exp(-math.log(2.0) / 20.0 * self.dt)
        self.grid_scanned *= decay
        prev_scanned_mass = float(np.sum(self.grid_p * self.grid_scanned))
        scanned_now = (np.hypot(xx - self.helico_x, yy - self.helico_y) <= self.radar_base_range).astype(np.float32)
        self.grid_scanned = np.maximum(self.grid_scanned, scanned_now)
        scanned_mass = float(np.sum(self.grid_p * self.grid_scanned))
        information_gain = max(0.0, scanned_mass - prev_scanned_mass)

        # Compute progress against the current, already-updated belief.
        unscanned_p = self.grid_p * (1.0 - self.grid_scanned)
        peak_j, peak_i = np.unravel_index(np.argmax(unscanned_p), unscanned_p.shape)
        peak_x, peak_y = xs[peak_i], ys[peak_j]
        progress = 5.0 * (math.hypot(peak_x - prev_x, peak_y - prev_y) - math.hypot(peak_x - self.helico_x, peak_y - self.helico_y))
        reward = -self.penalty_time * self.dt + self.reward_exploration * information_gain + progress

        dist_target = math.hypot(self.target_x - self.helico_x, self.target_y - self.helico_y)
        p_det = 0.0
        if dist_target <= self.radar_base_range:
            p_det = max(0.0, 0.98 * (1.0 - (dist_target / self.radar_base_range) ** 2))
        terminated = False
        truncated = False
        if dist_target < 0.8 or (p_det > 0.05 and rng.random() < p_det):
            self.intercepted = True
            reward += self.reward_detection
            terminated = True

        dist_frigate = math.hypot(self.helico_x - self.frigate_x, self.helico_y - self.frigate_y)
        time_to_frigate = dist_frigate / self.helico_max_speed * 60.0
        bingo_buffer = 15.0
        if self.fuel_remaining < time_to_frigate:
            self.bingo_failed = True
            reward -= self.penalty_bingo
            terminated = True
        elif self.fuel_remaining < time_to_frigate + bingo_buffer:
            heading_to_frigate = math.atan2(self.frigate_x - self.helico_x, self.frigate_y - self.helico_y)
            angle_diff = abs((self.helico_heading - heading_to_frigate + math.pi) % (2.0 * math.pi) - math.pi)
            reward -= 2.0 * angle_diff / math.pi

        hw, hh = self.search_area_width / 2.0, self.search_area_height / 2.0
        self.out_of_bounds = abs(self.helico_x) > hw or abs(self.helico_y) > hh
        if self.out_of_bounds:
            reward -= self.penalty_bingo
            terminated = True
        if self.fuel_remaining <= 0:
            truncated = True

        info = {
            "intercepted": self.intercepted,
            "bingo_failed": self.bingo_failed,
            "out_of_bounds": self.out_of_bounds,
            "target_bounced": self.target_bounced,
            "dist_to_target": dist_target,
            "fuel_remaining": self.fuel_remaining,
            "information_gain": information_gain,
        }
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info
