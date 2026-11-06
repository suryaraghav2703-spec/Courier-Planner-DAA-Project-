"""
Phase 1: Data Models & Problem Formulation for Mini-Courier Planner.

Defines structured dataclasses for Parcels, Delivery Nodes, Graph Edges,
Packing Plans, and Delivery Routes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import csv
from pathlib import Path


@dataclass
class Parcel:
    """
    Represents a customer parcel awaiting dispatch from the central courier hub.

    Attributes:
        parcel_id: Unique identifier (e.g., 'P001').
        customer_name: Recipient name or organization.
        destination_id: Graph node ID for delivery (e.g., 'C3').
        destination_name: Human-readable zone name.
        weight_kg: Integer weight in kilograms (used in 0/1 Knapsack DP).
        priority_tier: 1 = Critical/Express, 2 = Standard, 3 = Economy.
        deadline_hrs: Remaining hours until SLA delivery deadline.
        value_score: Priority-adjusted utility / delivery value score.
        fragile: Whether special handling is required.
    """
    parcel_id: str
    customer_name: str
    destination_id: str
    destination_name: str
    weight_kg: int
    priority_tier: int
    deadline_hrs: float
    value_score: int
    fragile: bool = False

    @property
    def value_density(self) -> float:
        """Value-to-weight ratio (score per kg)."""
        return round(self.value_score / max(1, self.weight_kg), 2)

    @property
    def priority_label(self) -> str:
        """Human-readable label for the priority tier."""
        labels = {1: "CRITICAL (T1)", 2: "STANDARD (T2)", 3: "ECONOMY  (T3)"}
        return labels.get(self.priority_tier, f"TIER-{self.priority_tier}")

    @property
    def composite_urgency_score(self) -> float:
        """
        Composite urgency bonus that rewards early deadlines and high priority tiers
        when combining prioritization with knapsack selection.
        """
        tier_bonus = {1: 30.0, 2: 15.0, 3: 0.0}.get(self.priority_tier, 0.0)
        deadline_bonus = max(0.0, (15.0 - self.deadline_hrs) * 2.0)
        return round(self.value_score + tier_bonus + deadline_bonus, 2)


@dataclass
class DeliveryNode:
    """Represents a depot hub or customer delivery zone in the road network."""
    node_id: str
    node_name: str
    zone_type: str
    x_km: float
    y_km: float


@dataclass
class RoadEdge:
    """Represents an undirected weighted road segment between two nodes."""
    source: str
    target: str
    distance_km: float
    road_name: str = ""


@dataclass
class PackingResult:
    """Stores the output of Phase 2 (0/1 Knapsack DP or Greedy baseline)."""
    algorithm_name: str
    capacity_kg: int
    selected_parcels: List[Parcel]
    deferred_parcels: List[Parcel]
    total_weight_kg: int
    total_value_score: int
    utilization_pct: float
    execution_time_ms: float
    dp_table: Optional[List[List[int]]] = field(default=None, repr=False)

    @property
    def unique_destinations(self) -> List[str]:
        """Ordered list of unique destination node IDs required by selected parcels."""
        seen = set()
        destinations = []
        for p in self.selected_parcels:
            if p.destination_id not in seen:
                seen.add(p.destination_id)
                destinations.append(p.destination_id)
        return destinations


@dataclass
class RouteLeg:
    """Represents one stop-to-stop transit leg in the courier delivery route."""
    leg_index: int
    from_node: str
    to_node: str
    shortest_distance_km: float
    intermediate_path: List[str]
    roads_taken: List[str]
    parcels_delivered: List[str]


@dataclass
class RouteResult:
    """Stores the output of Phase 3 (Dijkstra + Nearest-Neighbor / 2-Opt TSP)."""
    algorithm_name: str
    hub_id: str
    stop_sequence: List[str]
    expanded_road_path: List[str]
    legs: List[RouteLeg]
    total_distance_km: float
    dijkstra_time_ms: float
    tsp_time_ms: float
    distance_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict, repr=False)
    path_matrix: Dict[str, Dict[str, List[str]]] = field(default_factory=dict, repr=False)

    @property
    def total_execution_time_ms(self) -> float:
        return round(self.dijkstra_time_ms + self.tsp_time_ms, 4)


def load_parcels_from_csv(csv_path: str | Path) -> List[Parcel]:
    """Loads a list of Parcel objects from a CSV file."""
    parcels: List[Parcel] = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parcels.append(
                Parcel(
                    parcel_id=row["parcel_id"].strip(),
                    customer_name=row["customer_name"].strip(),
                    destination_id=row["destination_id"].strip(),
                    destination_name=row["destination_name"].strip(),
                    weight_kg=int(row["weight_kg"]),
                    priority_tier=int(row["priority_tier"]),
                    deadline_hrs=float(row["deadline_hrs"]),
                    value_score=int(row["value_score"]),
                    fragile=row.get("fragile", "False").strip().lower() == "true",
                )
            )
    return parcels


def load_nodes_from_csv(csv_path: str | Path) -> Dict[str, DeliveryNode]:
    """Loads delivery nodes from a CSV file keyed by node_id."""
    nodes: Dict[str, DeliveryNode] = {}
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node = DeliveryNode(
                node_id=row["node_id"].strip(),
                node_name=row["node_name"].strip(),
                zone_type=row["zone_type"].strip(),
                x_km=float(row["x_km"]),
                y_km=float(row["y_km"]),
            )
            nodes[node.node_id] = node
    return nodes


def load_edges_from_csv(csv_path: str | Path) -> List[RoadEdge]:
    """Loads road network edges from a CSV file."""
    edges: List[RoadEdge] = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges.append(
                RoadEdge(
                    source=row["source"].strip(),
                    target=row["target"].strip(),
                    distance_km=float(row["distance_km"]),
                    road_name=row.get("road_name", "").strip(),
                )
            )
    return edges
