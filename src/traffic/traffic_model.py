"""Traffic congestion model for realistic urban emergency dispatch simulation."""

from dataclasses import dataclass, field
import math
from typing import Any
import numpy as np

from src.config import VEHICLE_SPEED

# Default Metropolitan Bounding Box (San Francisco / Bay Area)
# Matches 100x100 simulation grid:
# Q0 (SW): Sunset / Twin Peaks / Ingleside
# Q1 (SE): Mission / Potrero / Bayview / I-280
# Q2 (NW): Presidio / Marina / Richmond
# Q3 (NE): Downtown / Financial District / SOMA / Bay Bridge
DEFAULT_METRO_BOUNDS = {
    "name": "San Francisco Metro",
    "lat_min": 37.7050,
    "lat_max": 37.8100,
    "lon_min": -122.5150,
    "lon_max": -122.3750,
    "center_lat": 37.7575,
    "center_lon": -122.4450,
    "zoom": 13,
}


def grid_to_latlon(
    x: float,
    y: float,
    bounds: dict[str, Any] = DEFAULT_METRO_BOUNDS,
) -> tuple[float, float]:
    """Convert simulation grid (0..100, 0..100) to real-world latitude and longitude."""
    lat = bounds["lat_min"] + (y / 100.0) * (bounds["lat_max"] - bounds["lat_min"])
    lon = bounds["lon_min"] + (x / 100.0) * (bounds["lon_max"] - bounds["lon_min"])
    return round(lat, 6), round(lon, 6)


def latlon_to_grid(
    lat: float,
    lon: float,
    bounds: dict[str, Any] = DEFAULT_METRO_BOUNDS,
) -> tuple[float, float]:
    """Convert real-world latitude and longitude back to simulation grid (0..100)."""
    lat_span = bounds["lat_max"] - bounds["lat_min"]
    lon_span = bounds["lon_max"] - bounds["lon_min"]
    y = ((lat - bounds["lat_min"]) / lat_span) * 100.0 if lat_span > 0 else 50.0
    x = ((lon - bounds["lon_min"]) / lon_span) * 100.0 if lon_span > 0 else 50.0
    return max(0.0, min(100.0, x)), max(0.0, min(100.0, y))


@dataclass
class TrafficZone:
    """Circular congestion hotspot (e.g. downtown gridlock, festival, construction)."""
    id: str
    name: str
    center_x: float
    center_y: float
    radius: float
    multiplier: float  # e.g. 2.5 = 2.5x slower (60% speed drop)
    description: str = ""

    def get_multiplier_at(self, x: float, y: float) -> float:
        """Return traffic multiplier at point with smooth Gaussian falloff."""
        dist = math.hypot(x - self.center_x, y - self.center_y)
        if dist >= self.radius:
            return 1.0
        # Smooth cosine/gaussian decay from center to edge
        decay = 0.5 * (1.0 + math.cos(math.pi * dist / self.radius))
        return 1.0 + (self.multiplier - 1.0) * decay


@dataclass
class TrafficCorridor:
    """Linear road corridor/artery with traffic condition."""
    id: str
    name: str
    x1: float
    y1: float
    x2: float
    y2: float
    multiplier: float  # 1.0 = clear highway, 3.0 = choked arterial
    width: float = 6.0
    is_expressway: bool = False

    def distance_to_segment(self, px: float, py: float) -> float:
        """Perpendicular distance from point to corridor line segment."""
        vx = self.x2 - self.x1
        vy = self.y2 - self.y1
        seg_len_sq = vx * vx + vy * vy
        if seg_len_sq < 1e-6:
            return math.hypot(px - self.x1, py - self.y1)
        # Projection fraction t
        t = max(0.0, min(1.0, ((px - self.x1) * vx + (py - self.y1) * vy) / seg_len_sq))
        proj_x = self.x1 + t * vx
        proj_y = self.y1 + t * vy
        return math.hypot(px - proj_x, py - proj_y)

    def get_multiplier_at(self, px: float, py: float) -> float:
        """Multiplier contribution near this corridor."""
        dist = self.distance_to_segment(px, py)
        if dist >= self.width:
            return 1.0
        decay = 1.0 - (dist / self.width)
        return 1.0 + (self.multiplier - 1.0) * decay


class TrafficModel:
    """Manages urban traffic congestion zones, arterial corridors, and path travel times."""

    def __init__(
        self,
        zones: list[TrafficZone] | None = None,
        corridors: list[TrafficCorridor] | None = None,
        base_speed: float = VEHICLE_SPEED,
        bounds: dict[str, Any] = DEFAULT_METRO_BOUNDS,
    ) -> None:
        self.zones = zones or []
        self.corridors = corridors or []
        self.base_speed = base_speed
        self.bounds = bounds

    def get_point_multiplier(self, x: float, y: float, t: float = 0.0) -> float:
        """Compute aggregate traffic multiplier at a specific grid coordinate."""
        mult = 1.0

        # Zone contributions
        for z in self.zones:
            zm = z.get_multiplier_at(x, y)
            if zm > mult:
                mult = zm

        # Corridor contributions
        for c in self.corridors:
            cm = c.get_multiplier_at(x, y)
            if cm > mult:
                mult = cm

        return min(4.5, max(1.0, mult))

    def get_traffic_multiplier(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        t: float = 0.0,
        samples: int = 12,
    ) -> float:
        """Numerically integrate traffic multiplier along direct path segment between two points."""
        if samples <= 1:
            return (self.get_point_multiplier(x1, y1, t) + self.get_point_multiplier(x2, y2, t)) / 2.0

        total = 0.0
        for i in range(samples):
            frac = (i + 0.5) / samples
            sx = x1 + frac * (x2 - x1)
            sy = y1 + frac * (y2 - y1)
            total += self.get_point_multiplier(sx, sy, t)

        avg_mult = total / samples
        return round(float(avg_mult), 2)

    def get_travel_time(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        t: float = 0.0,
    ) -> float:
        """Calculate effective travel time in minutes factoring in distance and traffic."""
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist < 1e-4:
            return 0.0
        mult = self.get_traffic_multiplier(x1, y1, x2, y2, t)
        return round(float((dist / self.base_speed) * mult), 2)

    def to_geojson(self) -> dict[str, Any]:
        """Export corridors and zones as GeoJSON for Leaflet real map rendering."""
        features = []

        # 1. Corridors as LineStrings
        for c in self.corridors:
            lat1, lon1 = grid_to_latlon(c.x1, c.y1, self.bounds)
            lat2, lon2 = grid_to_latlon(c.x2, c.y2, self.bounds)

            # Color coding by congestion level
            if c.multiplier >= 3.0:
                color = "#ef4444"  # severe red
                status = "GRIDLOCK"
            elif c.multiplier >= 2.0:
                color = "#f97316"  # heavy orange
                status = "HEAVY"
            elif c.multiplier > 1.2:
                color = "#eab308"  # moderate amber
                status = "MODERATE"
            else:
                color = "#10b981"  # free-flowing emerald
                status = "CLEAR"

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lon1, lat1], [lon2, lat2]],
                },
                "properties": {
                    "id": c.id,
                    "name": c.name,
                    "multiplier": c.multiplier,
                    "color": color,
                    "status": status,
                    "width": c.width,
                    "is_expressway": c.is_expressway,
                },
            })

        # 2. Zones as Point + Radius features
        for z in self.zones:
            lat, lon = grid_to_latlon(z.center_x, z.center_y, self.bounds)
            # Approx meters for radius
            lat_span_km = (self.bounds["lat_max"] - self.bounds["lat_min"]) * 111.0
            radius_meters = (z.radius / 100.0) * lat_span_km * 1000.0

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat],
                },
                "properties": {
                    "id": z.id,
                    "name": z.name,
                    "multiplier": z.multiplier,
                    "radius_meters": round(radius_meters, 0),
                    "description": z.description,
                    "type": "zone",
                },
            })

        return {
            "type": "FeatureCollection",
            "features": features,
            "bounds": self.bounds,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize traffic model state for JSON APIs."""
        return {
            "bounds": self.bounds,
            "zones": [
                {
                    "id": z.id,
                    "name": z.name,
                    "center_x": z.center_x,
                    "center_y": z.center_y,
                    "radius": z.radius,
                    "multiplier": z.multiplier,
                    "description": z.description,
                }
                for z in self.zones
            ],
            "corridors": [
                {
                    "id": c.id,
                    "name": c.name,
                    "x1": c.x1,
                    "y1": c.y1,
                    "x2": c.x2,
                    "y2": c.y2,
                    "multiplier": c.multiplier,
                    "width": c.width,
                    "is_expressway": c.is_expressway,
                }
                for c in self.corridors
            ],
            "geojson": self.to_geojson(),
        }


def get_default_traffic_model() -> TrafficModel:
    """Generate realistic metro traffic layout (San Francisco Metro preset).
    
    Includes:
    - Downtown Financial District severe gridlock (3.5x multiplier)
    - Market Street / Central Transit corridor (2.8x multiplier)
    - Bay Bridge Approach chokepoint (3.2x multiplier)
    - Clear Outer Ring Highway 101 / Sunset Expressway (1.0x - 1.1x clear bypass)
    """
    zones = [
        TrafficZone(
            id="Z_DOWNTOWN",
            name="Downtown & Financial District Gridlock",
            center_x=72.0,
            center_y=70.0,
            radius=18.0,
            multiplier=3.5,
            description="Peak congestion, dense high-rises and pedestrian crossings.",
        ),
        TrafficZone(
            id="Z_MARKET",
            name="Market Street Transit Chokepoint",
            center_x=62.0,
            center_y=58.0,
            radius=14.0,
            multiplier=2.8,
            description="Heavy traffic signal delays and bus lane restrictions.",
        ),
        TrafficZone(
            id="Z_BAY_BRIDGE",
            name="Bay Bridge & SOMA Bottleneck",
            center_x=84.0,
            center_y=64.0,
            radius=15.0,
            multiplier=3.2,
            description="Bridge approach queueing and interchange congestion.",
        ),
    ]

    corridors = [
        # Choked Corridors (Red)
        TrafficCorridor(
            id="C_MARKET_ST",
            name="Market Street Arterial",
            x1=50.0,
            y1=50.0,
            x2=78.0,
            y2=74.0,
            multiplier=3.4,
            width=5.0,
        ),
        TrafficCorridor(
            id="C_MISSION_ST",
            name="Mission Downtown Corridor",
            x1=52.0,
            y1=42.0,
            x2=80.0,
            y2=66.0,
            multiplier=2.9,
            width=5.0,
        ),
        TrafficCorridor(
            id="C_BAY_APPROACH",
            name="Bay Bridge Access",
            x1=70.0,
            y1=60.0,
            x2=95.0,
            y2=72.0,
            multiplier=3.2,
            width=6.0,
        ),
        # Moderate Arterials (Amber)
        TrafficCorridor(
            id="C_VAN_NESS",
            name="Van Ness Avenue (US-101)",
            x1=50.0,
            y1=20.0,
            x2=50.0,
            y2=85.0,
            multiplier=1.8,
            width=6.0,
        ),
        TrafficCorridor(
            id="C_GEARY_BLVD",
            name="Geary Boulevard East-West",
            x1=10.0,
            y1=72.0,
            x2=65.0,
            y2=72.0,
            multiplier=1.6,
            width=5.0,
        ),
        # Clear Bypass Expressways (Green)
        TrafficCorridor(
            id="C_HWY_101_SOUTH",
            name="Highway 101 Southbound Bypass",
            x1=45.0,
            y1=10.0,
            x2=85.0,
            y2=35.0,
            multiplier=1.0,
            width=7.0,
            is_expressway=True,
        ),
        TrafficCorridor(
            id="C_SUNSET_BLVD",
            name="Sunset Coastal Expressway",
            x1=12.0,
            y1=15.0,
            x2=12.0,
            y2=85.0,
            multiplier=1.05,
            width=7.0,
            is_expressway=True,
        ),
        TrafficCorridor(
            id="C_EMBARCADERO_BYPASS",
            name="Embarcadero Waterfront Arterial",
            x1=88.0,
            y1=45.0,
            x2=85.0,
            y2=92.0,
            multiplier=1.2,
            width=6.0,
            is_expressway=True,
        ),
    ]

    return TrafficModel(zones=zones, corridors=corridors)
