import math
import unittest

from navigation import advance_towards_waypoint, derive_initial_target_truth, solve_ground_track


class TestNavigation(unittest.TestCase):
    def test_headwind_reduces_ground_speed_without_changing_airspeed(self):
        solution = solve_ground_track(120.0, 90.0, 20.0, 90.0)
        self.assertAlmostEqual(solution["ground_speed"], 100.0)
        next_state = advance_towards_waypoint(
            x=0.0,
            y=0.0,
            waypoint_x=100.0,
            waypoint_y=0.0,
            airspeed=120.0,
            dt_minutes=1.0,
            wind_speed=20.0,
            wind_from_direction=90.0,
        )
        self.assertAlmostEqual(next_state["x"], 100.0 / 60.0)
        self.assertAlmostEqual(next_state["y"], 0.0)
        self.assertEqual(next_state["airspeed"], 120.0)

    def test_tailwind_increases_ground_speed(self):
        solution = solve_ground_track(120.0, 90.0, 20.0, 270.0)
        self.assertAlmostEqual(solution["ground_speed"], 140.0)

    def test_crosswind_crabs_and_maintains_track(self):
        solution = solve_ground_track(120.0, 90.0, 20.0, 0.0)
        self.assertLess(solution["air_heading"], 90.0)
        self.assertLess(solution["ground_speed"], 120.0)
        next_state = advance_towards_waypoint(
            x=0.0,
            y=0.0,
            waypoint_x=100.0,
            waypoint_y=0.0,
            airspeed=120.0,
            dt_minutes=1.0,
            wind_speed=20.0,
            wind_from_direction=0.0,
        )
        self.assertGreater(next_state["x"], 0.0)
        self.assertTrue(math.isclose(next_state["y"], 0.0, abs_tol=1e-9))

    def test_impossible_crosswind_does_not_fabricate_progress(self):
        solution = solve_ground_track(20.0, 90.0, 30.0, 0.0)
        self.assertFalse(solution["track_maintained"])
        self.assertEqual(solution["ground_speed"], 0.0)

    def test_initial_target_time_uncertainty_includes_current(self):
        truth = derive_initial_target_truth(
            datum_x=0.0,
            datum_y=0.0,
            spatial_offset_x=0.0,
            spatial_offset_y=0.0,
            time_offset_minutes=60.0,
            speed=10.0,
            heading=90.0,
            current_speed=2.0,
            current_heading=0.0,
        )
        self.assertAlmostEqual(truth["x"], 10.0)
        self.assertAlmostEqual(truth["y"], 2.0)


if __name__ == "__main__":
    unittest.main()
