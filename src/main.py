"""
Main Entry Point for Mini-Courier Planner Capstone Project.

Executes all 5 Capstone Phases end-to-end:
- Phase 1: Problem formulation, parcel & road graph ingestion, and architecture diagram.
- Phase 2: Parcel prioritization (Merge Sort vs. Quick/Heap/Insertion Sort) and
           box filling (0/1 Knapsack Dynamic Programming vs. 1D Optimized DP & Greedy).
- Phase 3: Customer graph adjacency representation, Dijkstra's shortest-path matrix,
           and Nearest-Neighbor TSP route construction (plus 2-Opt & Exact Held-Karp).
- Phase 4: Scalability benchmarking across Small, Medium, and Large datasets.
- Phase 5: Automated chart generation and terminal trace screenshot export in `images/`.
"""

from pathlib import Path
from typing import Dict, Any
from src.models import (
    load_parcels_from_csv,
    load_nodes_from_csv,
    load_edges_from_csv,
)
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
    format_dp_table_ascii,
)
from src.graph_routing import (
    CustomerGraph,
    plan_route_nearest_neighbor,
    plan_route_two_opt,
    plan_route_held_karp_exact,
)
from src.benchmark import (
    generate_synthetic_parcels,
    save_parcels_to_csv,
    run_sorting_benchmarks,
    run_knapsack_benchmarks,
    run_routing_benchmarks,
    run_end_to_end_dataset_comparison,
)
from src.visualizer import (
    plot_phase1_architecture,
    plot_phase2_knapsack_and_prioritization,
    plot_phase3_delivery_route,
    plot_phase4_runtime_charts,
    render_console_screenshot,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
IMAGES_DIR = PROJECT_ROOT / "images"


def execute_capstone_pipeline(van_capacity_kg: int = 30) -> Dict[str, Any]:
    """Runs the complete Mini-Courier Planner pipeline and returns all results."""
    # Ensure benchmark CSV datasets exist in datasets/
    save_parcels_to_csv(generate_synthetic_parcels(20, seed=520), DATASETS_DIR / "parcels_small.csv")
    save_parcels_to_csv(generate_synthetic_parcels(100, seed=600), DATASETS_DIR / "parcels_medium.csv")
    save_parcels_to_csv(generate_synthetic_parcels(500, seed=1000), DATASETS_DIR / "parcels_large.csv")

    # Load primary sample dataset
    parcels = load_parcels_from_csv(DATASETS_DIR / "parcels_sample.csv")
    nodes = load_nodes_from_csv(DATASETS_DIR / "graph_nodes.csv")
    edges = load_edges_from_csv(DATASETS_DIR / "graph_edges.csv")
    graph = CustomerGraph(nodes=nodes, edges=edges)

    # Phase 2A: Sorting
    merge_res = merge_sort_parcels(parcels, mode="deadline_priority")
    quick_res = quick_sort_parcels(parcels, mode="deadline_priority")
    heap_res = heap_sort_parcels(parcels, mode="deadline_priority")
    insert_res = insertion_sort_parcels(parcels, mode="deadline_priority")
    sorted_parcels = merge_res.sorted_parcels

    # Phase 2B: 0/1 Knapsack Dynamic Programming
    dp_2d_res = solve_knapsack_01_dp(sorted_parcels, capacity_kg=van_capacity_kg, store_table=True)
    dp_1d_res = solve_knapsack_1d_optimized(sorted_parcels, capacity_kg=van_capacity_kg)
    greedy_res = solve_knapsack_greedy_heuristic(sorted_parcels, capacity_kg=van_capacity_kg)

    # Phase 3: Graph Processing & Route Maker (Dijkstra + Nearest-Neighbor TSP)
    nn_route = plan_route_nearest_neighbor(graph, dp_2d_res.selected_parcels, hub_id="HUB")
    two_opt_route = plan_route_two_opt(graph, dp_2d_res.selected_parcels, hub_id="HUB")
    exact_route = plan_route_held_karp_exact(graph, dp_2d_res.selected_parcels, hub_id="HUB")

    # Phase 4: Scalability & Runtime Benchmarks
    sorting_bench = run_sorting_benchmarks()
    knapsack_bench = run_knapsack_benchmarks()
    routing_bench = run_routing_benchmarks()
    e2e_bench = run_end_to_end_dataset_comparison(graph)

    # Build Console Trace Report
    out = []
    out.append("=" * 94)
    out.append("              MINI-COURIER PLANNER — INTEGRATED ALGORITHMIC CAPSTONE SYSTEM")
    out.append("=" * 94)
    out.append(f"Loaded Sample Manifest : {len(parcels)} parcels | Total Manifest Weight: {sum(p.weight_kg for p in parcels)} kg")
    out.append(f"Courier Van Capacity   : W = {van_capacity_kg} kg | Road Network: |V| = {len(graph.nodes)} nodes, |E| = {len(graph.edges)} edges")
    out.append("")

    # Phase 2A Output
    out.append("PHASE 2A: PARCEL PRIORITIZATION (MERGE SORT BY PRIORITY TIER -> DEADLINE -> VALUE)")
    out.append("-" * 94)
    out.append(f"{'Rank':<5} {'ID':<6} {'Customer':<24} {'Dest':<18} {'Tier':<14} {'Deadline':<10} {'Wt(kg)':<7} {'Value':<6} {'Density'}")
    out.append("-" * 94)
    for idx, p in enumerate(sorted_parcels, start=1):
        dest_str = f"{p.destination_id} ({p.destination_name[:11]})"
        out.append(
            f"#{idx:<4d} {p.parcel_id:<6} {p.customer_name[:22]:<24} {dest_str:<18} "
            f"{p.priority_label:<14} {p.deadline_hrs:>5.1f} hrs  {p.weight_kg:>4d} kg  {p.value_score:>5d}  {p.value_density:>6.2f}"
        )
    out.append("-" * 94)
    out.append(
        f"Sorting Comparison (N={len(parcels)}): "
        f"MergeSort={merge_res.execution_time_ms:.4f}ms ({merge_res.comparisons_count} cmp) | "
        f"QuickSort={quick_res.execution_time_ms:.4f}ms ({quick_res.comparisons_count} cmp) | "
        f"HeapSort={heap_res.execution_time_ms:.4f}ms ({heap_res.comparisons_count} cmp) | "
        f"InsertionSort={insert_res.execution_time_ms:.4f}ms ({insert_res.comparisons_count} cmp)"
    )
    out.append("")

    # Phase 2B Output
    out.append(f"PHASE 2B: OPTIMAL BOX FILLING VIA 0/1 KNAPSACK DYNAMIC PROGRAMMING (CAPACITY W = {van_capacity_kg} KG)")
    out.append("-" * 94)
    out.append(format_dp_table_ascii(sorted_parcels, dp_2d_res.dp_table, step=3))
    out.append("-" * 94)
    sel_ids = ", ".join(f"{p.parcel_id}({p.weight_kg}kg,v={p.value_score}->{p.destination_id})" for p in dp_2d_res.selected_parcels)
    def_ids = ", ".join(f"{p.parcel_id}({p.weight_kg}kg)" for p in dp_2d_res.deferred_parcels)
    out.append(f"OPTIMAL DP SELECTED PARCELS ({len(dp_2d_res.selected_parcels)}): {sel_ids}")
    out.append(f"DEFERRED PARCELS ({len(dp_2d_res.deferred_parcels)})           : {def_ids}")
    out.append(
        f"0/1 Knapsack DP Result : Total Weight = {dp_2d_res.total_weight_kg}/{van_capacity_kg} kg "
        f"({dp_2d_res.utilization_pct:.1f}% util) | Optimal Value = {dp_2d_res.total_value_score} "
        f"| Runtime = {dp_2d_res.execution_time_ms:.4f} ms"
    )
    out.append(
        f"Greedy Baseline Result : Total Weight = {greedy_res.total_weight_kg}/{van_capacity_kg} kg "
        f"({greedy_res.utilization_pct:.1f}% util) | Greedy Value  = {greedy_res.total_value_score} "
        f"(DP Gain = +{dp_2d_res.total_value_score - greedy_res.total_value_score} value points, +{dp_2d_res.utilization_pct - greedy_res.utilization_pct:.1f}% capacity used)"
    )
    out.append("")

    # Phase 3 Output
    out.append("PHASE 3: CUSTOMER GRAPH, DIJKSTRA SHORTEST PATHS & NEAREST-NEIGHBOR TSP ROUTE")
    out.append("-" * 94)
    out.append(f"{'Leg':<4} {'From -> To':<12} {'Dist (km)':<10} {'Dijkstra Shortest Road Path':<28} {'Parcels Dropped Off'}")
    out.append("-" * 94)
    for leg in nn_route.legs:
        hop_str = " -> ".join(leg.intermediate_path)
        p_str = ", ".join(leg.parcels_delivered)
        out.append(
            f"#{leg.leg_index:<3d} {leg.from_node:>3} -> {leg.to_node:<4}  "
            f"{leg.shortest_distance_km:>6.2f} km   {hop_str:<28} {p_str}"
        )
    out.append("-" * 94)
    out.append(f"FINAL STOP SEQUENCE (NN TSP)   : {' -> '.join(nn_route.stop_sequence)}")
    out.append(f"FULL STREET-LEVEL NAVIGATION   : {' -> '.join(nn_route.expanded_road_path)}")
    out.append(
        f"Total Tour Length Comparison   : Nearest-Neighbor = {nn_route.total_distance_km:.2f} km | "
        f"2-Opt Refined = {two_opt_route.total_distance_km:.2f} km | "
        f"Held-Karp Exact Optimal = {exact_route.total_distance_km:.2f} km"
    )
    out.append("")

    # Phase 4 Output
    out.append("PHASE 4: SCALABILITY BENCHMARKS ACROSS SMALL, MEDIUM, AND LARGE DATASETS")
    out.append("-" * 94)
    out.append(f"{'Dataset Tier':<32} {'Packed':<10} {'Opt Value':<10} {'Sort(ms)':<10} {'DP(ms)':<10} {'Route(ms)':<10} {'Total(ms)'}")
    out.append("-" * 94)
    for row in e2e_bench:
        route_t = round(row["dijkstra_ms"] + row["tsp_ms"], 4)
        out.append(
            f"{row['dataset_label']:<32} "
            f"{row['selected_count']:>3d}/{row['n_parcels']:<5d} "
            f"{row['packed_value']:<10d} "
            f"{row['sort_ms']:<10.4f} "
            f"{row['knapsack_dp_ms']:<10.4f} "
            f"{route_t:<10.4f} "
            f"{row['total_pipeline_ms']:<10.4f}"
        )
    out.append("=" * 94)

    console_report = "\n".join(out)

    # Generate all 5 Phase visualizations in images/
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    plot_phase1_architecture(IMAGES_DIR / "phase1_system_architecture.png")
    plot_phase2_knapsack_and_prioritization(
        sorted_parcels, dp_2d_res, greedy_res, IMAGES_DIR / "phase2_knapsack_and_prioritization.png"
    )
    plot_phase3_delivery_route(
        graph, nn_route, two_opt_route, IMAGES_DIR / "phase3_delivery_route_map.png"
    )
    plot_phase4_runtime_charts(
        sorting_bench, knapsack_bench, routing_bench, e2e_bench, IMAGES_DIR / "phase4_runtime_scalability_charts.png"
    )
    render_console_screenshot(
        console_report, IMAGES_DIR / "phase5_console_execution_screenshot.png"
    )

    return {
        "parcels": parcels,
        "graph": graph,
        "merge_res": merge_res,
        "quick_res": quick_res,
        "heap_res": heap_res,
        "insert_res": insert_res,
        "dp_2d_res": dp_2d_res,
        "dp_1d_res": dp_1d_res,
        "greedy_res": greedy_res,
        "nn_route": nn_route,
        "two_opt_route": two_opt_route,
        "exact_route": exact_route,
        "sorting_bench": sorting_bench,
        "knapsack_bench": knapsack_bench,
        "routing_bench": routing_bench,
        "e2e_bench": e2e_bench,
        "console_report": console_report,
    }


if __name__ == "__main__":
    results = execute_capstone_pipeline(van_capacity_kg=30)
    print(results["console_report"])
