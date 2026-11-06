"""
Phase 4: Performance Testing, Scalability Benchmarking, and Dataset Generation.

Generates small, medium, and large synthetic courier datasets, measures execution
times across major algorithmic modules, and compiles structured benchmark results.
"""

import csv
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple
from src.models import DeliveryNode, Parcel, RoadEdge
from src.sorting import (
    merge_sort_parcels,
    quick_sort_parcels,
    heap_sort_parcels,
    insertion_sort_parcels,
)
from src.knapsack_dp import (
    solve_knapsack_01_dp,
    solve_knapsack_1d_optimized,
    solve_knapsack_greedy_heuristic,
)
from src.graph_routing import (
    CustomerGraph,
    plan_route_nearest_neighbor,
    plan_route_two_opt,
    plan_route_held_karp_exact,
)


ZONE_NAMES = [
    ("C1", "Downtown Market"),
    ("C2", "Civic Center"),
    ("C3", "Medical District"),
    ("C4", "Westside Mall"),
    ("C5", "Waterfront Plaza"),
    ("C6", "University Campus"),
    ("C7", "IT Corridor"),
    ("C8", "Industrial Park"),
    ("C9", "Uptown Boulevard"),
    ("C10", "Greenfield Suburb"),
]


def generate_synthetic_parcels(count: int, seed: int = 42, num_zones: int = 10) -> List[Parcel]:
    """Generates a deterministic synthetic parcel list of size `count`."""
    rng = random.Random(seed)
    parcels: List[Parcel] = []
    for i in range(1, count + 1):
        if num_zones <= len(ZONE_NAMES):
            dest_id, dest_name = rng.choice(ZONE_NAMES[:num_zones])
        else:
            z_idx = rng.randint(1, num_zones)
            dest_id = f"C{z_idx}"
            dest_name = f"Zone-{z_idx}"

        priority = rng.choices([1, 2, 3], weights=[0.30, 0.45, 0.25])[0]
        weight = rng.randint(1, 12)
        if priority == 1:
            deadline = round(rng.uniform(1.0, 4.0), 1)
            base_val = rng.randint(75, 145)
        elif priority == 2:
            deadline = round(rng.uniform(3.5, 8.5), 1)
            base_val = rng.randint(50, 110)
        else:
            deadline = round(rng.uniform(7.0, 16.0), 1)
            base_val = rng.randint(35, 85)

        parcels.append(
            Parcel(
                parcel_id=f"P{i:04d}",
                customer_name=f"Client-{i:04d}",
                destination_id=dest_id,
                destination_name=dest_name,
                weight_kg=weight,
                priority_tier=priority,
                deadline_hrs=deadline,
                value_score=base_val,
                fragile=(priority == 1 and weight <= 6),
            )
        )
    return parcels


def save_parcels_to_csv(parcels: List[Parcel], csv_path: str | Path) -> None:
    """Writes a list of Parcel objects to a CSV file."""
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "parcel_id", "customer_name", "destination_id", "destination_name",
            "weight_kg", "priority_tier", "deadline_hrs", "value_score", "fragile"
        ])
        for p in parcels:
            writer.writerow([
                p.parcel_id, p.customer_name, p.destination_id, p.destination_name,
                p.weight_kg, p.priority_tier, p.deadline_hrs, p.value_score, p.fragile
            ])


def generate_synthetic_graph(num_customers: int, seed: int = 42) -> CustomerGraph:
    """
    Generates a connected geometric city road network with 1 HUB + `num_customers` stops.
    Connects each node to its k nearest Euclidean neighbors with realistic road detour factors.
    """
    rng = random.Random(seed)
    nodes: Dict[str, DeliveryNode] = {
        "HUB": DeliveryNode("HUB", "Central Dispatch Hub", "Depot", 10.0, 10.0)
    }
    for i in range(1, num_customers + 1):
        angle = (2.0 * math.pi * i) / num_customers + rng.uniform(-0.15, 0.15)
        radius = rng.uniform(3.5, 9.5)
        x = round(10.0 + radius * math.cos(angle), 2)
        y = round(10.0 + radius * math.sin(angle), 2)
        nodes[f"C{i}"] = DeliveryNode(f"C{i}", f"Zone-{i}", "Customer", x, y)

    node_ids = list(nodes.keys())
    added_pairs = set()
    edges: List[RoadEdge] = []

    for u in node_ids:
        # Sort other nodes by Euclidean distance
        others = sorted(
            [v for v in node_ids if v != u],
            key=lambda v: math.hypot(nodes[u].x_km - nodes[v].x_km, nodes[u].y_km - nodes[v].y_km),
        )
        deg = min(4, len(others))
        for v in others[:deg]:
            pair = tuple(sorted((u, v)))
            if pair not in added_pairs:
                added_pairs.add(pair)
                euclid = math.hypot(nodes[u].x_km - nodes[v].x_km, nodes[u].y_km - nodes[v].y_km)
                road_dist = round(euclid * rng.uniform(1.08, 1.28), 2)
                edges.append(RoadEdge(u, v, max(1.0, road_dist), f"Rd-{u}-{v}"))

    return CustomerGraph(nodes=nodes, edges=edges)


def run_sorting_benchmarks(
    sizes: List[int] = [25, 50, 100, 250, 500, 1000, 1500]
) -> List[Dict[str, Any]]:
    """Benchmarks Merge Sort, Quick Sort, Heap Sort, and Insertion Sort across input sizes."""
    records: List[Dict[str, Any]] = []
    for n in sizes:
        parcels = generate_synthetic_parcels(n, seed=100 + n)
        ms = merge_sort_parcels(parcels)
        qs = quick_sort_parcels(parcels)
        hs = heap_sort_parcels(parcels)
        ins = insertion_sort_parcels(parcels)

        # Verify all algorithms produce identical key sequences
        assert [p.parcel_id for p in ms.sorted_parcels] == [
            p.parcel_id for p in ins.sorted_parcels
        ], "Stability / key mismatch between Merge Sort and Insertion Sort!"

        records.append({
            "n": n,
            "merge_sort_ms": ms.execution_time_ms,
            "quick_sort_ms": qs.execution_time_ms,
            "heap_sort_ms": hs.execution_time_ms,
            "insertion_sort_ms": ins.execution_time_ms,
            "merge_comparisons": ms.comparisons_count,
            "quick_comparisons": qs.comparisons_count,
            "heap_comparisons": hs.comparisons_count,
            "insertion_comparisons": ins.comparisons_count,
        })
    return records


def run_knapsack_benchmarks(
    configs: List[Tuple[int, int]] = [
        (20, 30),
        (50, 50),
        (100, 80),
        (250, 120),
        (500, 180),
        (1000, 250),
    ]
) -> List[Dict[str, Any]]:
    """Benchmarks 2D 0/1 Knapsack DP, 1D Optimized DP, and Greedy Heuristic."""
    records: List[Dict[str, Any]] = []
    for n, cap in configs:
        parcels = merge_sort_parcels(generate_synthetic_parcels(n, seed=200 + n)).sorted_parcels
        dp2d = solve_knapsack_01_dp(parcels, cap, store_table=False)
        dp1d = solve_knapsack_1d_optimized(parcels, cap)
        greedy = solve_knapsack_greedy_heuristic(parcels, cap)

        assert dp2d.total_value_score == dp1d.total_value_score, "DP 2D and 1D value mismatch!"

        records.append({
            "n": n,
            "capacity_kg": cap,
            "state_space_nW": n * cap,
            "dp_2d_ms": dp2d.execution_time_ms,
            "dp_1d_ms": dp1d.execution_time_ms,
            "greedy_ms": greedy.execution_time_ms,
            "dp_optimal_value": dp2d.total_value_score,
            "greedy_value": greedy.total_value_score,
            "value_gain_over_greedy": dp2d.total_value_score - greedy.total_value_score,
            "dp_utilization_pct": dp2d.utilization_pct,
            "greedy_utilization_pct": greedy.utilization_pct,
        })
    return records


def run_routing_benchmarks(
    stop_counts: List[int] = [5, 7, 9, 11, 15, 25, 40]
) -> List[Dict[str, Any]]:
    """Benchmarks Dijkstra + Nearest-Neighbor, 2-Opt, and Held-Karp Exact TSP."""
    records: List[Dict[str, Any]] = []
    for k in stop_counts:
        graph = generate_synthetic_graph(num_customers=k, seed=300 + k)
        # Create one parcel per customer stop so all k stops are active
        parcels = [
            Parcel(f"P{i:03d}", f"Client-{i}", f"C{i}", f"Zone-{i}", 2, 1, 2.0, 100)
            for i in range(1, k + 1)
        ]

        nn_res = plan_route_nearest_neighbor(graph, parcels)
        opt2_res = plan_route_two_opt(graph, parcels)

        exact_dist = None
        exact_ms = None
        if k <= 11:
            hk_res = plan_route_held_karp_exact(graph, parcels)
            exact_dist = hk_res.total_distance_km
            exact_ms = hk_res.tsp_time_ms

        records.append({
            "num_stops": k,
            "dijkstra_ms": nn_res.dijkstra_time_ms,
            "nn_tsp_ms": nn_res.tsp_time_ms,
            "two_opt_tsp_ms": opt2_res.tsp_time_ms,
            "held_karp_ms": exact_ms,
            "nn_distance_km": nn_res.total_distance_km,
            "two_opt_distance_km": opt2_res.total_distance_km,
            "exact_distance_km": exact_dist,
        })
    return records


def run_end_to_end_dataset_comparison(base_graph: CustomerGraph) -> List[Dict[str, Any]]:
    """
    Evaluates the complete Mini-Courier Planner pipeline across Small (N=20),
    Medium (N=100), and Large (N=500) parcel datasets.
    """
    tiers = [
        ("Small Dataset (N=20, W=35kg)", 20, 35, 10),
        ("Medium Dataset (N=100, W=80kg)", 100, 80, 10),
        ("Large Dataset (N=500, W=180kg)", 500, 180, 10),
    ]
    summary: List[Dict[str, Any]] = []
    for label, n, cap, zones in tiers:
        parcels = generate_synthetic_parcels(n, seed=500 + n, num_zones=zones)
        sort_res = merge_sort_parcels(parcels)
        pack_res = solve_knapsack_01_dp(sort_res.sorted_parcels, cap, store_table=False)
        route_res = plan_route_nearest_neighbor(base_graph, pack_res.selected_parcels)
        opt_route = plan_route_two_opt(base_graph, pack_res.selected_parcels)

        total_ms = (
            sort_res.execution_time_ms
            + pack_res.execution_time_ms
            + route_res.total_execution_time_ms
        )
        summary.append({
            "dataset_label": label,
            "n_parcels": n,
            "capacity_kg": cap,
            "selected_count": len(pack_res.selected_parcels),
            "packed_weight_kg": pack_res.total_weight_kg,
            "packed_value": pack_res.total_value_score,
            "route_stops": len(route_res.stop_sequence) - 2,
            "nn_route_km": route_res.total_distance_km,
            "two_opt_route_km": opt_route.total_distance_km,
            "sort_ms": sort_res.execution_time_ms,
            "knapsack_dp_ms": pack_res.execution_time_ms,
            "dijkstra_ms": route_res.dijkstra_time_ms,
            "tsp_ms": route_res.tsp_time_ms,
            "total_pipeline_ms": round(total_ms, 4),
        })
    return summary
