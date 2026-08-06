import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class BaysianPatrolEnv(gym.Env):
    """
    Gymnasium Environment for Maritime Search & Rescue Helicopter Patrol.
    Simulates Bayesian probability grid evolution, stochastic target motion,
    helicopter kinematics, radar probability of detection, and Bingo Fuel safety limits.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        grid_dim: int = 32,
        search_area_width: float = 120.0,
        search_area_height: float = 120.0,
        dt: float = 1.0,  # minutes per step
        max_endurance: float = 180.0,  # minutes
        helico_max_speed: float = 140.0,  # knots
        radar_base_range: float = 15.0,  # NM
        max_turn_rate_deg_min: float = 180.0,  # 180 deg/min
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
        
        # Correct max turn rate per step (180 deg/min * dt)
        self.max_turn_rate_rad_step = math.radians(max_turn_rate_deg_min) * dt
        
        # Adjustable Reward Parameters
        self.reward_detection = reward_detection
        self.reward_exploration = reward_exploration
        self.penalty_time = penalty_time
        self.penalty_bingo = penalty_bingo

        # Observation Space:
        # 1. 'grid': (2, grid_dim, grid_dim) -> Channel 0: Bayesian P(x,y,t), Channel 1: Scanned mask (0..1)
        # 2. 'vector': (10,) -> [x_norm, y_norm, heading_sin, heading_cos, speed_norm, fuel_ratio, dist_frigate_norm, wind_x_norm, wind_y_norm, radar_range_norm]
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(low=0.0, high=1.0, shape=(2, grid_dim, grid_dim), dtype=np.float32),
                "vector": spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32),
            }
        )

        # Action Space:
        # Continuous action in [-1, 1]^2:
        # action[0]: Normalized Delta Heading change (scaled to max turn rate)
        # action[1]: Normalized Delta Speed change (scaled to max speed acceleration)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self._reset_state()

    def _reset_state(self):
        # Randomize Frigate position anywhere in search area
        self.frigate_x = np.random.uniform(-0.35 * self.search_area_width, 0.35 * self.search_area_width)
        self.frigate_y = np.random.uniform(-0.35 * self.search_area_height, 0.35 * self.search_area_height)

        # Randomize target Datum position anywhere in search area
        self.datum_x = np.random.uniform(-0.35 * self.search_area_width, 0.35 * self.search_area_width)
        self.datum_y = np.random.uniform(-0.35 * self.search_area_height, 0.35 * self.search_area_height)

        self.target_x = self.datum_x + np.random.normal(0.0, 2.0)
        self.target_y = self.datum_y + np.random.normal(0.0, 2.0)
        self.target_speed = np.random.uniform(8.0, 22.0)  # kts
        self.target_heading = np.random.uniform(0, 2 * math.pi)

        # Helicopter initial state at frigate with random initial heading
        self.helico_x = self.frigate_x
        self.helico_y = self.frigate_y
        self.helico_heading = np.random.uniform(0, 2 * math.pi)
        self.helico_speed = 100.0
        self.fuel_remaining = self.max_endurance
        self.t = 0.0

        # Environmental wind
        self.wind_speed = np.random.uniform(5.0, 25.0)
        self.wind_dir = np.random.uniform(0, 360.0)

        # Grid setup
        self.grid_dim = 32
        self.grid_p = np.zeros((self.grid_dim, self.grid_dim), dtype=np.float32)
        self.grid_scanned = np.zeros((self.grid_dim, self.grid_dim), dtype=np.float32)

        # Initialize Gaussian probability blob around datum
        xs = np.linspace(-self.search_area_width / 2, self.search_area_width / 2, self.grid_dim)
        ys = np.linspace(-self.search_area_height / 2, self.search_area_height / 2, self.grid_dim)
        xx, yy = np.meshgrid(xs, ys)
        dist_sq = (xx - self.datum_x) ** 2 + (yy - self.datum_y) ** 2
        self.grid_p = np.exp(-dist_sq / (2 * (15.0 ** 2)))
        self.grid_p /= np.sum(self.grid_p) + 1e-8

        self.intercepted = False
        self.bingo_failed = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def _get_obs(self):
        grid_obs = np.stack([self.grid_p, self.grid_scanned], axis=0).astype(np.float32)

        x_norm = self.helico_x / (self.search_area_width / 2)
        y_norm = self.helico_y / (self.search_area_height / 2)
        heading_sin = math.sin(self.helico_heading)
        heading_cos = math.cos(self.helico_heading)
        speed_norm = self.helico_speed / self.helico_max_speed
        fuel_ratio = self.fuel_remaining / self.max_endurance

        dist_frigate = math.hypot(self.helico_x - self.frigate_x, self.helico_y - self.frigate_y)
        dist_frigate_norm = dist_frigate / self.search_area_width

        wind_rad = math.radians(self.wind_dir)
        wind_x_norm = (self.wind_speed * math.sin(wind_rad)) / 50.0
        wind_y_norm = (self.wind_speed * math.cos(wind_rad)) / 50.0
        radar_norm = self.radar_base_range / 30.0

        vector_obs = np.array(
            [
                x_norm,
                y_norm,
                heading_sin,
                heading_cos,
                speed_norm,
                fuel_ratio,
                dist_frigate_norm,
                wind_x_norm,
                wind_y_norm,
                radar_norm,
            ],
            dtype=np.float32,
        )

        return {"grid": grid_obs, "vector": vector_obs}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        delta_heading_req = action[0] * self.max_turn_rate_rad_step
        delta_speed_req = action[1] * (20.0 * self.dt / 60.0)

        prev_helico_x = self.helico_x
        prev_helico_y = self.helico_y

        # Update helicopter heading & speed
        self.helico_heading = (self.helico_heading + delta_heading_req) % (2 * math.pi)
        self.helico_speed = np.clip(self.helico_speed + delta_speed_req, 60.0, self.helico_max_speed)

        # Kinematic displacement over dt
        dist_step = (self.helico_speed / 60.0) * self.dt
        self.helico_x += dist_step * math.sin(self.helico_heading)
        self.helico_y += dist_step * math.cos(self.helico_heading)

        # Target displacement over dt
        target_dist_step = (self.target_speed / 60.0) * self.dt
        self.target_x += target_dist_step * math.sin(self.target_heading)
        self.target_y += target_dist_step * math.cos(self.target_heading)

        self.fuel_remaining -= self.dt
        self.t += self.dt

        # Find current peak probability cell location
        xs = np.linspace(-self.search_area_width / 2, self.search_area_width / 2, self.grid_dim)
        ys = np.linspace(-self.search_area_height / 2, self.search_area_height / 2, self.grid_dim)
        unscanned_p = self.grid_p * (1.0 - self.grid_scanned)
        if np.max(unscanned_p) > 0:
            peak_j, peak_i = np.unravel_index(np.argmax(unscanned_p), unscanned_p.shape)
            peak_x, peak_y = xs[peak_i], ys[peak_j]
        else:
            peak_x, peak_y = 0.0, 0.0

        prev_dist_to_peak = math.hypot(peak_x - prev_helico_x, peak_y - prev_helico_y)
        curr_dist_to_peak = math.hypot(peak_x - self.helico_x, peak_y - self.helico_y)
        dist_progress_reward = 5.0 * (prev_dist_to_peak - curr_dist_to_peak)

        # Check distance to target
        dist_to_target = math.hypot(self.target_x - self.helico_x, self.target_y - self.helico_y)

        # Calculate radar Pdet
        p_det = 0.0
        if dist_to_target <= self.radar_base_range:
            # Aspect & range dependent detection probability
            rel_range = dist_to_target / self.radar_base_range
            p_det = max(0.0, 0.98 * (1.0 - (rel_range ** 2)))

        # Dynamic Target Prior Distribution Drift (moving probability center over time)
        drift_x = self.datum_x + (self.target_speed / 60.0) * self.t * math.sin(self.target_heading)
        drift_y = self.datum_y + (self.target_speed / 60.0) * self.t * math.cos(self.target_heading)
        xx, yy = np.meshgrid(xs, ys)
        grid_p_unnorm = np.exp(-0.5 * (((xx - drift_x)**2) / 25.0 + ((yy - drift_y)**2) / 25.0))
        self.grid_p = grid_p_unnorm / np.sum(grid_p_unnorm)

        # Temporal Memory Relaxation on scanned mask (targets drift, so unscanned uncertainty flows back)
        decay_rate = math.exp((-math.log(2) / 20.0) * self.dt)
        self.grid_scanned *= decay_rate

        # Update grid scanning near helicopter
        prev_p_sum = np.sum(self.grid_p * self.grid_scanned)
        
        dists_to_helico = np.hypot(xx - self.helico_x, yy - self.helico_y)
        scanned_now = (dists_to_helico <= self.radar_base_range).astype(np.float32)
        self.grid_scanned = np.maximum(self.grid_scanned, scanned_now)

        new_p_sum = np.sum(self.grid_p * self.grid_scanned)
        prob_scanned_gain = max(0.0, new_p_sum - prev_p_sum)

        # Calculate rewards
        reward = -self.penalty_time * self.dt  # Time penalty
        reward += self.reward_exploration * prob_scanned_gain  # Information gain
        reward += dist_progress_reward  # Directional progress towards density peak
        
        # Calculate Bingo Distance & Margin
        dist_to_frigate = math.hypot(self.helico_x - self.frigate_x, self.helico_y - self.frigate_y)
        time_to_frigate = (dist_to_frigate / self.helico_max_speed) * 60.0  # in minutes
        bingo_buffer = 15.0  # minutes reserve

        # Rewards & Termination logic
        terminated = False
        truncated = False

        # Check Target Detection
        if dist_to_target < 0.8 or (p_det > 0.05 and np.random.random() < p_det):
            self.intercepted = True
            reward += self.reward_detection
            terminated = True

        # Check Bingo Fuel Safety
        if self.fuel_remaining < (time_to_frigate + bingo_buffer):
            # Helicopter crossed Bingo Fuel limit
            if self.fuel_remaining < time_to_frigate:
                # Out of fuel before frigate!
                reward -= self.penalty_bingo
                self.bingo_failed = True
                terminated = True
            else:
                # Return heading penalty encouragement
                heading_to_frigate = math.atan2(self.frigate_x - self.helico_x, self.frigate_y - self.helico_y)
                angle_diff = abs((self.helico_heading - heading_to_frigate + math.pi) % (2 * math.pi) - math.pi)
                reward -= 2.0 * (angle_diff / math.pi)

        # 5. Out of bounds penalty
        if (
            abs(self.helico_x) > self.search_area_width / 2
            or abs(self.helico_y) > self.search_area_height / 2
        ):
            reward -= 50.0

        if self.fuel_remaining <= 0:
            truncated = True

        info = {
            "intercepted": self.intercepted,
            "bingo_failed": self.bingo_failed,
            "dist_to_target": dist_to_target,
            "fuel_remaining": self.fuel_remaining,
        }

        return self._get_obs(), float(reward), terminated, truncated, info
