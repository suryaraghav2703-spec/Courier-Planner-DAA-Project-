"""
Visualization & Evidence Generation Module for Mini-Courier Planner.

Generates high-resolution charts and console execution screenshots in `images/`
covering all 5 Capstone Phases:
1. `images/phase1_system_architecture.png`
2. `images/phase2_knapsack_and_prioritization.png`
3. `images/phase3_delivery_route_map.png`
4. `images/phase4_runtime_scalability_charts.png`
5. `images/phase5_console_execution_screenshot.png`
"""

from pathlib import Path
from typing import Any, Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.models import PackingResult, Parcel, RouteResult
from src.graph_routing import CustomerGraph


def plot_phase1_architecture(output_path: str | Path) -> None:
    """Renders a clean Phase 1 System Architecture & Algorithmic Pipeline diagram."""
    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=180)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    fig.patch.set_facecolor("#F8FAFC")

    ax.text(
        6.5, 5.05,
        "Mini-Courier Planner: End-to-End Algorithmic Architecture (Phase 1 Design)",
        ha="center", va="center", fontsize=15, fontweight="bold", color="#0F172A"
    )
    ax.text(
        6.5, 4.65,
        "Integrating Sorting (Merge Sort), Dynamic Programming (0/1 Knapsack), Graph Shortest Paths (Dijkstra), & TSP Approximation",
        ha="center", va="center", fontsize=10, color="#475569"
    )

    stages = [
        (0.5, 1.4, 2.4, 2.7, "#1E293B", "#E2E8F0", "STAGE 1: INPUT DATA",
         "• Parcel Manifest CSV\n  (ID, Weight, Priority,\n   Deadline, Value Score)\n• City Road Graph CSV\n  (|V|=11 Nodes, |E|=22)\n• Van Capacity W = 30 kg"),
        (3.4, 1.4, 2.6, 2.7, "#0369A1", "#E0F2FE", "STAGE 2A: SORTING",
         "• Algorithm: Merge Sort\n• Complexity: O(n log n)\n• Property: Stable Sort\n• Multi-Key Comparator:\n  1. Priority Tier (ASC)\n  2. SLA Deadline (ASC)\n  3. Value Score (DESC)"),
        (6.5, 1.4, 2.8, 2.7, "#047857", "#D1FAE5", "STAGE 2B: 0/1 KNAPSACK DP",
         "• Algorithm: Bottom-Up DP\n• Complexity: O(n · W)\n• State: dp[i][w] Max Value\n• Backtracking selects\n  optimal parcel subset\n  fitting W <= 30 kg\n• Output: Packing Plan"),
        (9.8, 1.4, 2.7, 2.7, "#6D28D9", "#EDE9FE", "STAGE 3: GRAPH & TSP",
         "• Dijkstra All-Pairs Shortest\n  Paths: O(k·(V+E) log V)\n• Nearest-Neighbor TSP\n  Approximation: O(k²)\n• 2-Opt Local Refinement\n• Output: Ordered Tour"),
    ]

    for x, y, w, h, hdr_col, body_col, title, body in stages:
        rect_body = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
            facecolor=body_col, edgecolor=hdr_col, linewidth=2
        )
        ax.add_patch(rect_body)
        rect_hdr = mpatches.FancyBboxPatch(
            (x, y + h - 0.55), w, 0.55, boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=hdr_col, edgecolor=hdr_col, linewidth=1
        )
        ax.add_patch(rect_hdr)
        ax.text(x + w / 2, y + h - 0.27, title, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="white")
        ax.text(x + 0.14, y + h - 0.80, body, ha="left", va="top",
                fontsize=8.7, color="#0F172A", linespacing=1.42, family="monospace")

    # Arrows between boxes
    arrow_xs = [(2.98, 3.32), (6.08, 6.42), (9.38, 9.72)]
    for x0, x1 in arrow_xs:
        ax.annotate(
            "", xy=(x1, 2.75), xytext=(x0, 2.75),
            arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.35", lw=2.5, color="#334155")
        )

    # Bottom banner for Phase 4 & 5
    banner = mpatches.FancyBboxPatch(
        (0.5, 0.35), 12.0, 0.72, boxstyle="round,pad=0.06,rounding_size=0.12",
        facecolor="#FEF3C7", edgecolor="#D97706", linewidth=1.8
    )
    ax.add_patch(banner)
    ax.text(
        6.5, 0.71,
        "PHASE 4 & 5 EVALUATION: Empirical Scalability Benchmarks (Small N=20, Medium N=100, Large N=500) + Complexity Proofs + Visual Reports",
        ha="center", va="center", fontsize=9.5, fontweight="bold", color="#92400E"
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_phase2_knapsack_and_prioritization(
    sorted_parcels: List[Parcel],
    packing_dp: PackingResult,
    packing_greedy: PackingResult,
    output_path: str | Path,
) -> None:
    """
    Renders Phase 2 visual evidence:
    - Panel A: Prioritized Parcels (Merge Sort order) colored by DP Selection vs. Deferred.
    - Panel B: 2D 0/1 Knapsack Dynamic Programming Table Heatmap with Backtracking Path.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.8), dpi=180, gridspec_kw={"width_ratios": [1.05, 1.35]})
    fig.patch.set_facecolor("#FFFFFF")

    selected_ids = {p.parcel_id for p in packing_dp.selected_parcels}

    # --- Panel A: Prioritized Parcels & Box-Filling Status ---
    labels = [
        f"#{idx+1:02d} {p.parcel_id} (T{p.priority_tier}, {p.deadline_hrs}h, {p.weight_kg}kg -> {p.destination_id})"
        for idx, p in enumerate(sorted_parcels)
    ]
    values = [p.value_score for p in sorted_parcels]
    colors = ["#10B981" if p.parcel_id in selected_ids else "#94A3B8" for p in sorted_parcels]

    y_pos = np.arange(len(sorted_parcels))
    bars = ax1.barh(y_pos, values, color=colors, edgecolor="#1E293B", height=0.72, linewidth=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=8.5, family="monospace")
    ax1.invert_yaxis()
    ax1.set_xlabel("Parcel Priority Value Score", fontsize=10, fontweight="bold")
    ax1.set_title(
        f"Phase 2A & 2B: Merge-Sorted Parcels & Optimal Box Selection\n"
        f"(DP Value = {packing_dp.total_value_score} vs. Greedy = {packing_greedy.total_value_score} | "
        f"Packed Weight = {packing_dp.total_weight_kg}/{packing_dp.capacity_kg} kg)",
        fontsize=10.5, fontweight="bold", pad=10
    )
    ax1.set_xlim(0, max(values) * 1.38)
    ax1.set_ylim(len(sorted_parcels) + 1.1, -0.8)
    ax1.grid(axis="x", linestyle="--", alpha=0.4)

    for bar, p in zip(bars, sorted_parcels):
        status = "PACKED" if p.parcel_id in selected_ids else "DEFERRED"
        ax1.text(
            bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
            f"v={p.value_score} ({status})",
            va="center", ha="left", fontsize=7.8,
            fontweight="bold" if status == "PACKED" else "normal",
            color="#065F46" if status == "PACKED" else "#475569",
        )

    packed_patch = mpatches.Patch(color="#10B981", label=f"Selected by 0/1 DP ({len(packing_dp.selected_parcels)} parcels, {packing_dp.total_weight_kg} kg)")
    deferred_patch = mpatches.Patch(color="#94A3B8", label=f"Deferred to Next Trip ({len(packing_dp.deferred_parcels)} parcels)")
    ax1.legend(handles=[packed_patch, deferred_patch], loc="lower right", fontsize=8.5)

    # --- Panel B: 0/1 Knapsack DP Table Heatmap ---
    dp_matrix = np.array(packing_dp.dp_table, dtype=float)
    im = ax2.imshow(dp_matrix, aspect="auto", cmap="YlGnBu", origin="upper")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.042, pad=0.03)
    cbar.set_label("Cumulative Optimal Value Score dp[i][w]", fontsize=9.5, fontweight="bold")

    n_items, max_w_plus_1 = dp_matrix.shape
    row_labels = ["0: Empty"] + [f"{i}: {p.parcel_id} ({p.weight_kg}kg)" for i, p in enumerate(sorted_parcels, 1)]
    ax2.set_yticks(np.arange(n_items))
    ax2.set_yticklabels(row_labels, fontsize=8, family="monospace")
    col_ticks = np.arange(0, max_w_plus_1, 2)
    ax2.set_xticks(col_ticks)
    ax2.set_xticklabels([f"{w}" for w in col_ticks], fontsize=8)
    ax2.set_xlabel("Van Capacity w (kg)", fontsize=10, fontweight="bold")
    ax2.set_title(
        f"Phase 2B: 0/1 Knapsack DP State Table dp[i][w] (15 Parcels × {max_w_plus_1-1} kg)\n"
        f"Red Trajectory Shows Optimal Backtracking Path from dp[15][30] = {int(dp_matrix[-1, -1])}",
        fontsize=10.5, fontweight="bold", pad=10
    )

    # Trace backtracking path on heatmap
    rem_w = max_w_plus_1 - 1
    path_w = [rem_w]
    path_i = [n_items - 1]
    for i in range(n_items - 1, 0, -1):
        if dp_matrix[i, rem_w] != dp_matrix[i - 1, rem_w]:
            # Annotate selected transition cell
            ax2.scatter([rem_w], [i], color="#EF4444", s=65, zorder=6, edgecolors="white", linewidth=1.2)
            rem_w -= sorted_parcels[i - 1].weight_kg
        path_w.append(rem_w)
        path_i.append(i - 1)

    ax2.plot(path_w, path_i, color="#EF4444", linewidth=2.2, linestyle="-", label="DP Backtracking Trajectory", zorder=5)
    # Annotate sampled cell values
    for i in range(n_items):
        for w in range(0, max_w_plus_1, 5):
            val = int(dp_matrix[i, w])
            txt_col = "white" if val > dp_matrix.max() * 0.58 else "#0F172A"
            ax2.text(w, i, f"{val}", ha="center", va="center", fontsize=6.5, color=txt_col)
        # Also annotate final column W=30
        w_last = max_w_plus_1 - 1
        if w_last % 5 != 0:
            val = int(dp_matrix[i, w_last])
            txt_col = "white" if val > dp_matrix.max() * 0.58 else "#0F172A"
            ax2.text(w_last, i, f"{val}", ha="center", va="center", fontsize=6.5, color=txt_col, fontweight="bold")

    ax2.legend(loc="lower left", fontsize=8.5)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_phase3_delivery_route(
    graph: CustomerGraph,
    nn_route: RouteResult,
    two_opt_route: RouteResult,
    output_path: str | Path,
) -> None:
    """
    Renders Phase 3 visual evidence:
    - Panel A: City Road Network with Dijkstra shortest paths and Nearest-Neighbor TSP route.
    - Panel B: Dijkstra Pairwise Distance Matrix Heatmap & Tour Comparison (Nearest-Neighbor vs 2-Opt).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.8), dpi=180, gridspec_kw={"width_ratios": [1.25, 1.0]})
    fig.patch.set_facecolor("#FFFFFF")

    # --- Panel A: City Graph & Active Delivery Route ---
    active_stops = set(nn_route.stop_sequence)

    # Draw all road network edges in light gray
    for edge in graph.edges:
        u_node = graph.nodes[edge.source]
        v_node = graph.nodes[edge.target]
        ax1.plot(
            [u_node.x_km, v_node.x_km],
            [u_node.y_km, v_node.y_km],
            color="#CBD5E1", linewidth=1.6, linestyle="--", zorder=1
        )
        mx = (u_node.x_km + v_node.x_km) / 2.0
        my = (u_node.y_km + v_node.y_km) / 2.0
        ax1.text(
            mx, my, f"{edge.distance_km:.1f}km",
            fontsize=7, color="#64748B", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.75),
            zorder=2
        )

    # Highlight the physical road segments traversed by the Nearest-Neighbor TSP route
    leg_colors = plt.cm.plasma(np.linspace(0.12, 0.85, len(nn_route.legs)))
    for leg, col in zip(nn_route.legs, leg_colors):
        path_nodes = leg.intermediate_path
        for idx in range(len(path_nodes) - 1):
            n1 = graph.nodes[path_nodes[idx]]
            n2 = graph.nodes[path_nodes[idx + 1]]
            ax1.annotate(
                "",
                xy=(n2.x_km, n2.y_km),
                xytext=(n1.x_km, n1.y_km),
                arrowprops=dict(
                    arrowstyle="-|>,head_width=0.38,head_length=0.55",
                    color=col, lw=3.0, shrinkA=10, shrinkB=10
                ),
                zorder=4,
            )

    # Draw nodes
    stop_order_map: Dict[str, int] = {}
    for order_idx, node_id in enumerate(nn_route.stop_sequence[:-1]):
        stop_order_map[node_id] = order_idx

    for node_id, node in graph.nodes.items():
        if node_id == "HUB":
            ax1.scatter([node.x_km], [node.y_km], s=380, color="#DC2626", edgecolors="#1E293B", linewidth=2.0, marker="s", zorder=6)
            ax1.text(node.x_km, node.y_km - 1.05, f"HUB: {node.node_name}\n(Start / Return)",
                     ha="center", va="top", fontsize=8.5, fontweight="bold", color="#991B1B",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEF2F2", edgecolor="#DC2626", lw=1))
        elif node_id in active_stops:
            order_num = stop_order_map[node_id]
            ax1.scatter([node.x_km], [node.y_km], s=310, color="#0284C7", edgecolors="#0F172A", linewidth=1.8, zorder=6)
            ax1.text(node.x_km, node.y_km, f"#{order_num}", ha="center", va="center",
                     fontsize=8.5, fontweight="bold", color="white", zorder=7)
            ax1.text(node.x_km, node.y_km + 0.85, f"{node_id}: {node.node_name}",
                     ha="center", va="bottom", fontsize=8, fontweight="bold", color="#0F172A",
                     bbox=dict(boxstyle="round,pad=0.15", facecolor="#E0F2FE", edgecolor="#0284C7", lw=0.8))
        else:
            ax1.scatter([node.x_km], [node.y_km], s=180, color="#E2E8F0", edgecolors="#64748B", linewidth=1.2, zorder=5)
            ax1.text(node.x_km, node.y_km + 0.7, f"{node_id} (No Packed Parcel)",
                     ha="center", va="bottom", fontsize=7.5, color="#64748B")

    tour_str = " -> ".join(nn_route.stop_sequence)
    ax1.set_title(
        f"Phase 3: Customer Road Graph & Nearest-Neighbor TSP Delivery Route\n"
        f"Stop Sequence: {tour_str} (Total = {nn_route.total_distance_km:.2f} km)",
        fontsize=10.2, fontweight="bold", pad=10
    )
    ax1.set_xlabel("East-West Coordinate (km)", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("North-South Coordinate (km)", fontsize=9.5, fontweight="bold")
    ax1.set_xlim(0.5, 21.0)
    ax1.set_ylim(1.5, 19.5)
    ax1.grid(True, linestyle=":", alpha=0.45)

    # --- Panel B: Dijkstra Shortest-Path Distance Matrix Heatmap ---
    nodes_ordered = nn_route.stop_sequence[:-1]
    k = len(nodes_ordered)
    mat = np.zeros((k, k))
    for i, u in enumerate(nodes_ordered):
        for j, v in enumerate(nodes_ordered):
            mat[i, j] = nn_route.distance_matrix[u][v]

    im2 = ax2.imshow(mat, cmap="PuBu", aspect="auto")
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Shortest Path Distance (km)", fontsize=9, fontweight="bold")

    ax2.set_xticks(np.arange(k))
    ax2.set_yticks(np.arange(k))
    ax2.set_xticklabels(nodes_ordered, fontsize=9, fontweight="bold")
    ax2.set_yticklabels(nodes_ordered, fontsize=9, fontweight="bold")
    ax2.set_title(
        f"Phase 3: All-Pairs Dijkstra Shortest-Path Matrix (km)\n"
        f"NN Tour = {nn_route.total_distance_km:.2f} km vs. 2-Opt Refined = {two_opt_route.total_distance_km:.2f} km",
        fontsize=10.2, fontweight="bold", pad=10
    )

    for i in range(k):
        for j in range(k):
            is_tour_step = (j == (i + 1) % k)
            val = mat[i, j]
            col = "white" if val > mat.max() * 0.55 else "#0F172A"
            box = dict(boxstyle="round,pad=0.18", facecolor="none", edgecolor="#EF4444", lw=2.0) if is_tour_step else None
            ax2.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8.5,
                     fontweight="bold" if is_tour_step else "normal", color=col, bbox=box)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_phase4_runtime_charts(
    sorting_records: List[Dict[str, Any]],
    knapsack_records: List[Dict[str, Any]],
    routing_records: List[Dict[str, Any]],
    e2e_records: List[Dict[str, Any]],
    output_path: str | Path,
) -> None:
    """
    Renders a 4-panel Phase 4 Scalability & Runtime Comparison Dashboard.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), dpi=180)
    fig.patch.set_facecolor("#FFFFFF")
    ax1, ax2, ax3, ax4 = axes.flatten()

    # --- Panel 1: Sorting Runtime Scalability ---
    ns = [r["n"] for r in sorting_records]
    ax1.plot(ns, [r["merge_sort_ms"] for r in sorting_records], "o-", lw=2.4, color="#0284C7", label="Merge Sort O(n log n) [Selected]")
    ax1.plot(ns, [r["quick_sort_ms"] for r in sorting_records], "s--", lw=2.0, color="#10B981", label="Quick Sort O(n log n) Avg")
    ax1.plot(ns, [r["heap_sort_ms"] for r in sorting_records], "^-.", lw=2.0, color="#8B5CF6", label="Heap Sort O(n log n)")
    ax1.plot(ns, [r["insertion_sort_ms"] for r in sorting_records], "d:", lw=2.2, color="#EF4444", label="Insertion Sort O(n²) Baseline")
    ax1.set_title("Phase 4A: Sorting Algorithm Runtime vs. Parcel Count (N)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Number of Parcels (N)", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Execution Time (ms, log scale)", fontsize=9.5, fontweight="bold")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)
    ax1.legend(fontsize=8.5)

    # --- Panel 2: Knapsack DP vs 1D Optimized DP vs Greedy ---
    k_labels = [f"N={r['n']}\nW={r['capacity_kg']}" for r in knapsack_records]
    x_k = np.arange(len(k_labels))
    width = 0.26
    ax2.bar(x_k - width, [r["dp_2d_ms"] for r in knapsack_records], width, color="#0284C7", label="2D 0/1 Knapsack DP O(n·W)")
    ax2.bar(x_k, [r["dp_1d_ms"] for r in knapsack_records], width, color="#10B981", label="1D Space-Optimized DP O(n·W)")
    ax2.bar(x_k + width, [r["greedy_ms"] for r in knapsack_records], width, color="#F59E0B", label="Greedy Heuristic O(n log n)")
    ax2.set_xticks(x_k)
    ax2.set_xticklabels(k_labels, fontsize=8.5)
    ax2.set_yscale("log")
    ax2.set_title("Phase 4B: Knapsack Packing Runtime across Problem Sizes (N, W)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Problem Scale (Parcels N, Capacity W kg)", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Execution Time (ms, log scale)", fontsize=9.5, fontweight="bold")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax2.legend(fontsize=8.5)

    # --- Panel 3: Routing & TSP Runtime Scaling ---
    stops = [r["num_stops"] for r in routing_records]
    ax3.plot(stops, [r["dijkstra_ms"] for r in routing_records], "o-", lw=2.2, color="#0284C7", label="Dijkstra All-Pairs Shortest Paths")
    ax3.plot(stops, [r["nn_tsp_ms"] for r in routing_records], "s-", lw=2.2, color="#10B981", label="Nearest-Neighbor TSP O(k²)")
    ax3.plot(stops, [r["two_opt_tsp_ms"] for r in routing_records], "^--", lw=2.0, color="#F59E0B", label="NN + 2-Opt Local Search")

    hk_stops = [r["num_stops"] for r in routing_records if r["held_karp_ms"] is not None]
    hk_times = [r["held_karp_ms"] for r in routing_records if r["held_karp_ms"] is not None]
    ax3.plot(hk_stops, hk_times, "d-", lw=2.4, color="#EF4444", label="Held-Karp Exact TSP O(k²·2^k)")
    ax3.set_yscale("log")
    ax3.set_title("Phase 4C: Graph & TSP Routing Runtime vs. Active Stops (k)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Number of Active Customer Delivery Stops (k)", fontsize=9.5, fontweight="bold")
    ax3.set_ylabel("Execution Time (ms, log scale)", fontsize=9.5, fontweight="bold")
    ax3.grid(True, which="both", linestyle="--", alpha=0.4)
    ax3.legend(fontsize=8.5)

    # --- Panel 4: End-to-End Pipeline Breakdown (Small, Medium, Large) ---
    ds_labels = [r["dataset_label"].split("(")[0].strip() + f"\n(N={r['n_parcels']}, W={r['capacity_kg']}kg)" for r in e2e_records]
    x_e = np.arange(len(ds_labels))
    sort_t = np.array([r["sort_ms"] for r in e2e_records])
    dp_t = np.array([r["knapsack_dp_ms"] for r in e2e_records])
    dijk_t = np.array([r["dijkstra_ms"] for r in e2e_records])
    tsp_t = np.array([r["tsp_ms"] for r in e2e_records])

    b1 = ax4.bar(x_e, sort_t, width=0.48, color="#38BDF8", label="Phase 2A: Merge Sort")
    b2 = ax4.bar(x_e, dp_t, width=0.48, bottom=sort_t, color="#10B981", label="Phase 2B: 0/1 Knapsack DP")
    b3 = ax4.bar(x_e, dijk_t, width=0.48, bottom=sort_t + dp_t, color="#8B5CF6", label="Phase 3A: Dijkstra Shortest Paths")
    b4 = ax4.bar(x_e, tsp_t, width=0.48, bottom=sort_t + dp_t + dijk_t, color="#F97316", label="Phase 3B: Nearest-Neighbor TSP")

    totals = sort_t + dp_t + dijk_t + tsp_t
    for idx, tot in enumerate(totals):
        ax4.text(idx, tot + max(totals) * 0.025, f"Total: {tot:.2f} ms", ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color="#0F172A")

    ax4.set_xticks(x_e)
    ax4.set_xticklabels(ds_labels, fontsize=9, fontweight="bold")
    ax4.set_ylim(0, max(totals) * 1.18)
    ax4.set_title("Phase 4D: End-to-End Pipeline Runtime Breakdown by Dataset Scale", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Cumulative Execution Time (ms)", fontsize=9.5, fontweight="bold")
    ax4.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax4.legend(fontsize=8.5, loc="upper left")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def render_console_screenshot(console_text: str, output_path: str | Path) -> None:
    """
    Renders the terminal execution output trace as a clean macOS-style dark terminal
    screenshot PNG in `images/phase5_console_execution_screenshot.png`.
    """
    lines = console_text.strip("\n").splitlines()
    # Keep up to 78 lines for crisp readability
    if len(lines) > 78:
        lines = lines[:78]

    fig_height = max(9.5, len(lines) * 0.19 + 1.2)
    fig, ax = plt.subplots(figsize=(14.5, fig_height), dpi=180)
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, len(lines) + 4)
    ax.axis("off")

    # Terminal window header bar
    header_y = len(lines) + 2.3
    ax.add_patch(mpatches.Rectangle((0, header_y - 0.8), 100, 2.5, color="#1E293B"))
    ax.scatter([2.2, 4.0, 5.8], [header_y + 0.35] * 3, s=95, color=["#EF4444", "#F59E0B", "#10B981"], zorder=5)
    ax.text(
        50, header_y + 0.35,
        "zsh — python3 -m src.main — Mini-Courier Planner Capstone Execution Trace",
        ha="center", va="center", fontsize=9.5, fontweight="bold", color="#CBD5E1", family="monospace"
    )

    for idx, line in enumerate(lines):
        y = len(lines) - idx + 0.5
        color = "#E2E8F0"
        weight = "normal"
        if line.startswith("===") or line.startswith("PHASE"):
            color = "#38BDF8"
            weight = "bold"
        elif "OPTIMAL" in line or "FINAL" in line or "SELECTED" in line or "Total Tour" in line:
            color = "#34D399"
            weight = "bold"
        elif line.startswith("---") or line.startswith("Item"):
            color = "#94A3B8"
        ax.text(1.5, y, line, ha="left", va="center", fontsize=8.0, color=color, fontweight=weight, family="monospace")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.savefig(output_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
