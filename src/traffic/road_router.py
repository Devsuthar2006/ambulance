"""Real-world road network routing engine for emergency fleet navigation.

Fetches turn-by-turn road geometries (OSRM) and provides realistic urban street grid
turn-by-turn fallbacks with sub-second caching and polyline interpolation.
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from typing import Any

from src.traffic.traffic_model import DEFAULT_METRO_BOUNDS, grid_to_latlon, latlon_to_grid


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points in meters."""
    r = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def compute_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate forward compass bearing from point 1 to point 2 in degrees (0..360)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    y = math.sin(dlam) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    deg = math.degrees(math.atan2(y, x))
    return (deg + 360.0) % 360.0


def interpolate_road_polyline(
    waypoints: list[tuple[float, float]],
    fraction: float,
) -> tuple[float, float, float]:
    """Interpolate along a road polyline.

    Args:
        waypoints: List of (latitude, longitude) tuples along the road.
        fraction: Value between 0.0 (start) and 1.0 (destination).

    Returns:
        tuple of (current_lat, current_lon, heading_degrees).
    """
    if not waypoints:
        return 37.7575, -122.4450, 0.0
    if len(waypoints) == 1 or fraction <= 0.0:
        return waypoints[0][0], waypoints[0][1], 0.0
    if fraction >= 1.0:
        p_prev = waypoints[-2]
        p_last = waypoints[-1]
        heading = compute_bearing(p_prev[0], p_prev[1], p_last[0], p_last[1])
        return p_last[0], p_last[1], heading

    # Compute cumulative segment lengths
    seg_lengths: list[float] = []
    total_len = 0.0
    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i + 1]
        dist = haversine_distance(p1[0], p1[1], p2[0], p2[1])
        seg_lengths.append(dist)
        total_len += dist

    if total_len <= 1e-4:
        return waypoints[0][0], waypoints[0][1], 0.0

    target_dist = fraction * total_len
    acc = 0.0

    for i, length in enumerate(seg_lengths):
        if acc + length >= target_dist or i == len(seg_lengths) - 1:
            seg_frac = (target_dist - acc) / length if length > 1e-4 else 0.0
            seg_frac = max(0.0, min(1.0, seg_frac))
            p1 = waypoints[i]
            p2 = waypoints[i + 1]
            cur_lat = p1[0] + seg_frac * (p2[0] - p1[0])
            cur_lon = p1[1] + seg_frac * (p2[1] - p1[1])
            heading = compute_bearing(p1[0], p1[1], p2[0], p2[1])
            return cur_lat, cur_lon, heading
        acc += length

    last = waypoints[-1]
    return last[0], last[1], 0.0


def generate_street_grid_route(
    start_lat: float,
    start_lon: float,
    dest_lat: float,
    dest_lon: float,
) -> list[tuple[float, float]]:
    """Synthesize a realistic urban street-grid route with city turns and avenues.

    Used when offline or during OSRM latency spikes to ensure ambulances always
    follow city street blocks instead of cutting diagonals through buildings.
    """
    pts: list[tuple[float, float]] = [(start_lat, start_lon)]

    # Urban street grid block spacing (~100 meters)
    lat_step = 0.0010  # ~110m
    lon_step = 0.0012  # ~100m

    dlat = dest_lat - start_lat
    dlon = dest_lon - start_lon

    # Market Street diagonal corridor approximation for San Francisco Metro
    # (Angle ~45 deg between lat 37.75 and 37.79, lon -122.43 and -122.39)
    in_market_corridor = (
        min(start_lat, dest_lat) >= 37.74
        and max(start_lat, dest_lat) <= 37.80
        and min(start_lon, dest_lon) >= -122.44
        and max(start_lon, dest_lon) <= -122.39
    )

    if in_market_corridor and abs(dlat) > 0.01 and abs(dlon) > 0.01:
        # 3-segment arterial path: avenue -> arterial diagonal -> cross-street
        mid_lat = start_lat + 0.5 * dlat
        mid_lon = start_lon + 0.5 * dlon
        pts.append((round(start_lat, 6), round(mid_lon, 6)))
        pts.append((round(mid_lat, 6), round(mid_lon, 6)))
        pts.append((round(mid_lat, 6), round(dest_lon, 6)))
        pts.append((round(dest_lat, 6), round(dest_lon, 6)))
    else:
        # Multi-corner Manhattan grid navigation
        # Intermediate corner 1: follow current street/avenue to alignment
        steps = 4
        cur_la, cur_lo = start_lat, start_lon
        for s in range(1, steps):
            frac = s / float(steps)
            if s % 2 == 1:
                cur_lo = start_lon + frac * dlon
            else:
                cur_la = start_lat + frac * dlat
            pts.append((round(cur_la, 6), round(cur_lo, 6)))

        pts.append((round(dest_lat, 6), round(dest_lon, 6)))

    return pts


class RoadRouter:
    """Road Network Routing Service with in-memory caching and OSRM integration."""

    def __init__(self, bounds: dict[str, Any] = DEFAULT_METRO_BOUNDS) -> None:
        self.bounds = bounds
        self._cache: dict[str, list[tuple[float, float]]] = {}

    def _make_key(self, lat1: float, lon1: float, lat2: float, lon2: float) -> str:
        return f"{round(lat1, 4)},{round(lon1, 4)}->{round(lat2, 4)},{round(lon2, 4)}"

    def get_route_between_grid_points(
        self,
        start_x: float,
        start_y: float,
        dest_x: float,
        dest_y: float,
        use_osrm: bool = True,
    ) -> list[tuple[float, float]]:
        """Get road waypoints for simulation grid coordinates."""
        lat1, lon1 = grid_to_latlon(start_x, start_y, self.bounds)
        lat2, lon2 = grid_to_latlon(dest_x, dest_y, self.bounds)
        return self.get_route(lat1, lon1, lat2, lon2, use_osrm=use_osrm)

    def get_route(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        use_osrm: bool = True,
    ) -> list[tuple[float, float]]:
        """Return list of (lat, lon) road waypoints connecting the two points."""
        key = self._make_key(lat1, lon1, lat2, lon2)
        if key in self._cache:
            return self._cache[key]

        # Check trivial distance
        dist_direct = haversine_distance(lat1, lon1, lat2, lon2)
        if dist_direct < 10.0:
            route = [(lat1, lon1), (lat2, lon2)]
            self._cache[key] = route
            return route

        if use_osrm:
            try:
                # OSRM expects coordinates as {longitude},{latitude}
                url = (
                    f"https://router.project-osrm.org/route/v1/driving/"
                    f"{lon1:.6f},{lat1:.6f};{lon2:.6f},{lat2:.6f}"
                    f"?overview=full&geometries=geojson"
                )
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "RESQAI-EmergencyDispatcher/2.0 (Academic/Sim)"},
                )
                with urllib.request.urlopen(req, timeout=1.8) as resp:
                    if resp.status == 200:
                        payload = json.loads(resp.read().decode("utf-8"))
                        if payload.get("code") == "Ok" and payload.get("routes"):
                            geom_coords = payload["routes"][0]["geometry"]["coordinates"]
                            # Convert GeoJSON [lon, lat] -> [lat, lon]
                            route = [(round(float(c[1]), 6), round(float(c[0]), 6)) for c in geom_coords]
                            if len(route) >= 2:
                                self._cache[key] = route
                                return route
            except Exception:
                # Network latency, timeout, or offline — fallback seamlessly
                pass

        # High-fidelity urban street grid fallback
        route = generate_street_grid_route(lat1, lon1, lat2, lon2)
        self._cache[key] = route
        return route


# Singleton instance
_GLOBAL_ROUTER: RoadRouter | None = None


def get_default_road_router() -> RoadRouter:
    """Return the global RoadRouter singleton instance."""
    global _GLOBAL_ROUTER
    if _GLOBAL_ROUTER is None:
        _GLOBAL_ROUTER = RoadRouter()
    return _GLOBAL_ROUTER
