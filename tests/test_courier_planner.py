"""
Comprehensive Unit & Integration Test Suite for Mini-Courier Planner.

Verifies:
1. Phase 1: Dataset loaders and domain model properties.
2. Phase 2A: Sorting correctness, stability, and multi-key comparator ordering across
             Merge Sort, Quick Sort, Heap Sort, and Insertion Sort.
3. Phase 2B: 0/1 Knapsack Dynamic Programming optimality (2D & 1D) and superiority
             over Greedy heuristic on indivisible capacities.
4. Phase 3: CustomerGraph adjacency representation, Dijkstra's shortest paths,
            Nearest-Neighbor TSP tour validity, 2-Opt refinement, and Held-Karp exact DP.
"""

from pathlib import Path
import unittest

from src.models import (
    Parcel,
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
)
from src.graph_routing import (
    CustomerGraph,
    plan_route_nearest_neighbor,
    plan_route_two_opt,
    plan_route_held_karp_exact,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


class TestMiniCourierPlanner(unittest.TestCase):
    def setUp(self) -> None:
        self.parcels = load_parcels_from_csv(DATASETS_DIR / "parcels_sample.csv")
        self.nodes = load_nodes_from_csv(DATASETS_DIR / "graph_nodes.csv")
        self.edges = load_edges_from_csv(DATASETS_DIR / "graph_edges.csv")
        self.graph = CustomerGraph(nodes=self.nodes, edges=self.edges)

    def test_phase1_data_loading(self) -> None:
        self.assertEqual(len(self.parcels), 15)
        self.assertEqual(len(self.nodes), 11)
        self.assertEqual(len(self.edges), 22)
        self.assertIn("HUB", self.nodes)

    def test_phase2a_sorting_algorithms_consistency(self) -> None:
        ms = merge_sort_parcels(self.parcels)
        qs = quick_sort_parcels(self.parcels)
        hs = heap_sort_parcels(self.parcels)
        ins = insertion_sort_parcels(self.parcels)

        expected_ids = [p.parcel_id for p in ms.sorted_parcels]
        self.assertEqual([p.parcel_id for p in qs.sorted_parcels], expected_ids)
        self.assertEqual([p.parcel_id for p in hs.sorted_parcels], expected_ids)
        self.assertEqual([p.parcel_id for p in ins.sorted_parcels], expected_ids)

        # Verify monotonic non-decreasing (priority_tier, deadline_hrs, -value_score)
        for i in range(len(ms.sorted_parcels) - 1):
            a = ms.sorted_parcels[i]
            b = ms.sorted_parcels[i + 1]
            self.assertLessEqual(
                (a.priority_tier, a.deadline_hrs, -a.value_score),
                (b.priority_tier, b.deadline_hrs, -b.value_score),
            )

    def test_phase2b_knapsack_dp_optimality(self) -> None:
        sorted_parcels = merge_sort_parcels(self.parcels).sorted_parcels
        dp2d = solve_knapsack_01_dp(sorted_parcels, capacity_kg=30, store_table=True)
        dp1d = solve_knapsack_1d_optimized(sorted_parcels, capacity_kg=30)
        greedy = solve_knapsack_greedy_heuristic(sorted_parcels, capacity_kg=30)

        self.assertEqual(dp2d.total_value_score, 785)
        self.assertEqual(dp2d.total_weight_kg, 30)
        self.assertEqual(dp1d.total_value_score, dp2d.total_value_score)
        self.assertEqual(
            [p.parcel_id for p in dp1d.selected_parcels],
            [p.parcel_id for p in dp2d.selected_parcels],
        )
        # Verify DP strictly beats Greedy on the sample manifest
        self.assertGreater(dp2d.total_value_score, greedy.total_value_score)

    def test_phase3_dijkstra_and_tsp_routing(self) -> None:
        sorted_parcels = merge_sort_parcels(self.parcels).sorted_parcels
        dp2d = solve_knapsack_01_dp(sorted_parcels, capacity_kg=30)

        nn_route = plan_route_nearest_neighbor(self.graph, dp2d.selected_parcels)
        two_opt_route = plan_route_two_opt(self.graph, dp2d.selected_parcels)
        exact_route = plan_route_held_karp_exact(self.graph, dp2d.selected_parcels)

        # Start and end at HUB
        self.assertEqual(nn_route.stop_sequence[0], "HUB")
        self.assertEqual(nn_route.stop_sequence[-1], "HUB")

        # All unique destinations of selected parcels are visited
        required_stops = set(dp2d.unique_destinations)
        visited_stops = set(nn_route.stop_sequence[1:-1])
        self.assertEqual(visited_stops, required_stops)

        # Verify 2-Opt is <= Nearest-Neighbor and >= Exact Held-Karp
        self.assertLessEqual(two_opt_route.total_distance_km, nn_route.total_distance_km)
        self.assertGreaterEqual(two_opt_route.total_distance_km, exact_route.total_distance_km)
        self.assertAlmostEqual(two_opt_route.total_distance_km, 51.20, places=2)
        self.assertAlmostEqual(nn_route.total_distance_km, 52.60, places=2)


if __name__ == "__main__":
    unittest.main()
