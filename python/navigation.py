"""Wind-aware helicopter ground-track navigation primitives."""
from __future__ import annotations

import math


def solve_ground_track(
    airspeed: float,
    track_heading: float,
    wind_speed: float,
    wind_from_direction: float,
) -> dict[str, float]:
    track_rad = math.radians(track_heading)
    wind_to_rad = math.radians(wind_from_direction + 180.0)
    track_x, track_y = math.sin(track_rad), math.cos(track_rad)
    right_x, right_y = math.cos(track_rad), -math.sin(track_rad)
    wind_x = wind_speed * math.sin(wind_to_rad)
    wind_y = wind_speed * math.cos(wind_to_rad)
    along_wind = wind_x * track_x + wind_y * track_y
    cross_wind = wind_x * right_x + wind_y * right_y
    if abs(cross_wind) > airspeed:
        air_x = -right_x * math.copysign(airspeed, cross_wind)
        air_y = -right_y * math.copysign(airspeed, cross_wind)
        return {
            "ground_speed": 0.0,
            "air_heading": math.degrees(math.atan2(air_x, air_y)) % 360.0,
            "track_maintained": False,
        }
    air_along_track = math.sqrt(max(0.0, airspeed * airspeed - cross_wind * cross_wind))
    air_x = track_x * air_along_track - right_x * cross_wind
    air_y = track_y * air_along_track - right_y * cross_wind
    air_heading = math.degrees(math.atan2(air_x, air_y)) % 360.0
    return {
        "ground_speed": max(0.0, air_along_track + along_wind),
        "air_heading": air_heading,
        "track_maintained": True,
    }


def derive_initial_target_truth(
    *,
    datum_x: float,
    datum_y: float,
    spatial_offset_x: float,
    spatial_offset_y: float,
    time_offset_minutes: float,
    speed: float,
    heading: float,
    current_speed: float = 0.0,
    current_heading: float = 0.0,
) -> dict[str, float]:
    time_hours = time_offset_minutes / 60.0
    target_rad = math.radians(heading)
    current_rad = math.radians(current_heading)
    return {
        "x": datum_x + spatial_offset_x + (
            speed * math.sin(target_rad) + current_speed * math.sin(current_rad)
        ) * time_hours,
        "y": datum_y + spatial_offset_y + (
            speed * math.cos(target_rad) + current_speed * math.cos(current_rad)
        ) * time_hours,
    }


def advance_towards_waypoint(
    *,
    x: float,
    y: float,
    waypoint_x: float,
    waypoint_y: float,
    airspeed: float,
    dt_minutes: float,
    wind_speed: float = 0.0,
    wind_from_direction: float = 0.0,
) -> dict[str, float]:
    dx, dy = waypoint_x - x, waypoint_y - y
    remaining = math.hypot(dx, dy)
    if remaining <= 1e-9:
        return {"x": x, "y": y, "airspeed": airspeed, "air_heading": 0.0, "ground_speed": 0.0}
    track_heading = math.degrees(math.atan2(dx, dy)) % 360.0
    solution = solve_ground_track(airspeed, track_heading, wind_speed, wind_from_direction)
    travel = min(remaining, solution["ground_speed"] / 60.0 * dt_minutes)
    track_rad = math.radians(track_heading)
    return {
        "x": x + travel * math.sin(track_rad),
        "y": y + travel * math.cos(track_rad),
        "airspeed": airspeed,
        "air_heading": solution["air_heading"],
        "ground_speed": solution["ground_speed"],
    }


def estimate_travel_minutes(
    *,
    x: float,
    y: float,
    waypoint_x: float,
    waypoint_y: float,
    airspeed: float,
    wind_speed: float = 0.0,
    wind_from_direction: float = 0.0,
) -> float:
    dx, dy = waypoint_x - x, waypoint_y - y
    distance = math.hypot(dx, dy)
    if distance <= 1e-9:
        return 0.0
    track_heading = math.degrees(math.atan2(dx, dy)) % 360.0
    ground_speed = solve_ground_track(
        airspeed,
        track_heading,
        wind_speed,
        wind_from_direction,
    )["ground_speed"]
    return distance / ground_speed * 60.0 if ground_speed > 1e-9 else math.inf
