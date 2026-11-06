# Technical Project Report: Mini-Courier Planner
### Integrated Parcel Prioritization, 0/1 Knapsack Dynamic Programming, Dijkstra Shortest Paths, and TSP Route Approximation

- **Course Outcomes Mapped:** CO1, CO2, CO3, CO4, CO5 (L5 – Evaluate)
- **Total Marks:** 100 (Phases 1–5)
- **Version Tag:** `v1.0-final-submission`

---

## 1. Phase 1: Courier Problem Design and Planning (20 Marks)

### 1.1 Problem Statement & Real-World Motivation
Modern urban courier hubs process hundreds of heterogeneous parcels under strict delivery Service-Level Agreements (SLAs) and physical vehicle payload limits. Solving dispatch planning in isolation leads to sub-optimal operations: sorting alone ignores vehicle weight limits, knapsack packing alone ignores road network topology, and routing without prioritization risks delaying critical medical or express shipments.

The **Mini-Courier Planner** integrates four foundational algorithmic paradigms into a unified four-stage pipeline:
1. **Divide-and-Conquer Sorting (`Merge Sort`):** Prioritizes parcels by urgency tier, remaining SLA deadline hours, and value score.
2. **Dynamic Programming (`0/1 Knapsack DP`):** Selects the highest-value combination of indivisible parcels that fits within the courier van's payload capacity $W$.
3. **Greedy Graph Shortest Paths (`Dijkstra's Algorithm`):** Computes exact shortest-path distances and multi-hop street trajectories across the city road network.
4. **Combinatorial Approximation & Local Search (`Nearest-Neighbor TSP` & `2-Opt`):** Orders the selected customer stops into a low-cost closed delivery tour starting and ending at the central dispatch hub.

### 1.2 Data Design & Domain Attributes
The application defines clean, strongly typed dataclasses in `src/models.py`:
- **`Parcel` Attributes:**
  - `parcel_id` (`str`): Unique manifest identifier (`P001`–`P015`).
  - `customer_name` (`str`): Recipient organization name.
  - `destination_id` (`str`) & `destination_name` (`str`): Target node ID in the road network (`C1`–`C10`).
  - `weight_kg` (`int`): Integer parcel weight in kilograms ($w_i \in \mathbb{Z}^+$).
  - `priority_tier` (`int`): `1` = Critical/Express, `2` = Standard, `3` = Economy.
  - `deadline_hrs` (`float`): Remaining hours before SLA breach.
  - `value_score` (`int`): Priority-weighted dispatch utility score ($v_i \in \mathbb{Z}^+$).
  - `fragile` (`bool`): Special handling flag.
- **`CustomerGraph` Format:**
  - Represented as a **Weighted Undirected Adjacency List** `adj[u] = [(v, distance_km, road_name), ...]`.
  - Contains $|V| = 11$ nodes (`HUB` + 10 delivery zones `C1`–`C10`) and $|E| = 22$ road segments.
  - **Why Adjacency List over Adjacency Matrix?** Road networks are planar and sparse ($|E| \ll |V|^2$). An adjacency list uses $O(V + E)$ memory and iterates over incident edges in $O(\text{deg}(u))$ time, reducing Dijkstra's complexity from $O(V^2)$ to $O((V + E)\log V)$.

![Phase 1 Architecture](images/phase1_system_architecture.png)

---

## 2. Phase 2: Package Prioritization and Box Filling (20 Marks)

### 2.1 Parcel Prioritization via Merge Sort
Parcels are ordered using the composite lexicographical key:
$$\text{Key}(p_i) = (\text{priority\_tier}_i \uparrow,\; \text{deadline\_hrs}_i \uparrow,\; -\text{value\_score}_i \uparrow,\; \text{weight\_kg}_i \uparrow)$$

#### Justification of Merge Sort over Alternative Sorting Algorithms
| Sorting Algorithm | Best Time | Average Time | Worst Time | Aux Space | Stable? | Suitability for Courier Manifests |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Merge Sort (Selected)** | $\Theta(n \log n)$ | $\Theta(n \log n)$ | $\Theta(n \log n)$ | $O(n)$ | **Yes** | **Optimal choice:** Guarantees predictable $O(n \log n)$ latency and preserves arrival order (FIFO fairness) on identical keys. |
| **Quick Sort (Median-of-3)** | $\Theta(n \log n)$ | $\Theta(n \log n)$ | $O(n^2)$ | $O(\log n)$ | No | Fast in-place cache performance, but lacks stability and risks $O(n^2)$ worst-case degradation. |
| **Heap Sort** | $\Theta(n \log n)$ | $\Theta(n \log n)$ | $\Theta(n \log n)$ | $O(1)$ | No | Guaranteed $O(n \log n)$ worst-case, but unstable and performs ~70% more comparisons due to sift-down operations. |
| **Insertion Sort** | $\Theta(n)$ | $\Theta(n^2)$ | $\Theta(n^2)$ | $O(1)$ | Yes | Only viable for tiny manifests ($n < 20$); degrades quadratically at scale ($30\times$ slower at $n=1,500$). |

### 2.2 Optimal Box Filling via 0/1 Knapsack Dynamic Programming
Because physical courier parcels cannot be cut into fractions, package selection is modeled as the **0/1 Knapsack Problem** with van payload capacity $W = 30\text{ kg}$:
- **Subproblem State:** Let $dp[i][w]$ denote the maximum total value score achievable using a subset of the first $i$ prioritized parcels within capacity $w \in \{0, 1, \dots, W\}$.
- **Bellman Recurrence Relation:**
  $$dp[i][w] = \begin{cases} dp[i-1][w] & \text{if } w_i > w \\ \max\big(dp[i-1][w],\; v_i + dp[i-1][w - w_i]\big) & \text{if } w_i \le w \end{cases}$$
- **Selected Packing Plan ($W = 30\text{ kg}$):**
  - **Selected Parcels (8 items):** `P013` (6kg, v=140, C3), `P004` (2kg, v=85, C2), `P001` (4kg, v=95, C3), `P008` (3kg, v=90, C2), `P005` (3kg, v=80, C5), `P014` (2kg, v=105, C9), `P002` (6kg, v=114, C7), `P003` (4kg, v=76, C1).
  - **Total Weight Packed:** **$30 / 30\text{ kg}$ ($100.0\%$ capacity utilization)**.
  - **Total Priority Value Score:** **$785$** (packs **100% of Critical Tier-1 parcels** plus `P003`).
  - **Why 0/1 DP Beats Greedy Heuristic:** Sorting by value-to-weight ratio ($v_i/w_i$) causes the Greedy heuristic to pick `P010` ($7\text{ kg}, v=135$, density $19.29$) after the first 6 items ($20\text{ kg}$), leaving $3\text{ kg}$ of stranded capacity that no remaining parcel can fill ($27/30\text{ kg}$, value $= 730$). **0/1 Knapsack DP replaces `P010` (`7kg`) with `P002` (`6kg, v=114`) + `P003` (`4kg, v=76`), gaining $+55$ value points and achieving $100\%$ utilization.**

![Phase 2 Prioritization and Knapsack DP](images/phase2_knapsack_and_prioritization.png)

---

## 3. Phase 3: Route Maker and Graph Processing (20 Marks)

### 3.1 Individual Shortest Paths via Dijkstra's Algorithm
For the active delivery stops $U = \{\text{HUB}, C1, C2, C3, C5, C7, C9\}$ required by the 8 packed parcels, we run Dijkstra's single-source shortest path algorithm from each $u \in U$ using a binary min-heap priority queue (`heapq`):
- **Time Complexity:** $O(|U| \cdot (V + E)\log V)$.
- **Multi-Hop Shortest Path Discovery:** Several stop-to-stop transitions traverse intermediate transit nodes because multi-hop arterial links are shorter than direct avenues:
  - `C5 -> C7`: Shortest path is **`C5 -> C6 -> C7`** ($5.3 + 5.8 = 11.10\text{ km}$).
  - `C7 -> HUB`: Shortest path is **`C7 -> C6 -> HUB`** ($5.8 + 5.3 = 11.10\text{ km}$).

### 3.2 Overall Delivery Route via Nearest-Neighbor TSP & 2-Opt Refinement
Starting at `HUB`, the **Nearest-Neighbor TSP heuristic** iteratively selects the closest unvisited stop $v^* = \arg\min_{v \in U_{\text{rem}}} D[\text{curr}][v]$:

| Leg | From $\to$ To | Shortest Distance (km) | Street-Level Dijkstra Path | Roads Traversed | Parcels Delivered |
| :---: | :---: | :---: | :--- | :--- | :--- |
| **#1** | `HUB -> C1` | 4.80 km | `HUB -> C1` | Central Ave North | `P003 (4kg)` |
| **#2** | `C1 -> C9` | 4.20 km | `C1 -> C9` | Uptown Parkway | `P014 (2kg)` |
| **#3** | `C9 -> C2` | 7.80 km | `C9 -> C2` | North Ridge Ave | `P004 (2kg), P008 (3kg)` |
| **#4** | `C2 -> C3` | 4.70 km | `C2 -> C3` | West Civic Blvd | `P013 (6kg), P001 (4kg)` |
| **#5** | `C3 -> C5` | 8.90 km | `C3 -> C5` | Riverfront Bypass | `P005 (3kg)` |
| **#6** | `C5 -> C7` | 11.10 km | `C5 -> C6 -> C7` | South Bay Rd $\to$ Tech Valley Hwy | `P002 (6kg)` |
| **#7** | `C7 -> HUB` | 11.10 km | `C7 -> C6 -> HUB` | Tech Valley Hwy $\to$ Campus Link Rd | *Return to Dispatch Hub* |
| **Total** | **Closed Tour** | **52.60 km** | **9 Road Hops** | — | **8 Parcels (30 kg)** |

- **2-Opt & Held-Karp Exact Comparison:** Applying **2-Opt Local Search** uncrosses the `C9 -> C2` transition to yield `HUB -> C2 -> C3 -> C5 -> C7 -> C9 -> C1 -> HUB` with a total distance of **51.20 km**, which matches the **Held-Karp Exact TSP DP** global optimum (**saving 1.40 km / 2.66%**).

![Phase 3 Delivery Route Map](images/phase3_delivery_route_map.png)

---

## 4. Phase 4: Performance Testing, Complexity Analysis, and Optimization (20 Marks)

### 4.1 Asymptotic Complexity Summary

| Algorithmic Component | Implemented Method | Time Complexity | Space Complexity | Key Scalability Characteristic |
| :--- | :--- | :---: | :---: | :--- |
| **Parcel Sorting** | Merge Sort | $\Theta(n \log n)$ | $O(n)$ | Deterministic log-linear growth; stable ordering. |
| **Box Filling (Exact)** | 2D 0/1 Knapsack DP | $\Theta(n \cdot W)$ | $\Theta(n \cdot W)$ | Pseudo-polynomial; exact global optimum. |
| **Box Filling (Optimized)** | 1D Bitmask Knapsack DP | $\Theta(n \cdot W)$ | $\Theta(W)$ | Eliminates 2D table allocations; ~15–20% faster. |
| **Shortest Paths** | Min-Heap Dijkstra (Active Closure) | $O(k(V + E)\log V)$ | $O(V + k^2)$ | Scales near-linearly on sparse road graphs. |
| **Route Construction** | Nearest-Neighbor TSP | $\Theta(k^2)$ | $O(k)$ | Sub-millisecond even for $k = 40+$ stops. |
| **Route Refinement** | 2-Opt Local Search | $O(k^2)$ per pass | $O(k)$ | Closes ~95–100% of NN optimality gap in $<0.5\text{ ms}$. |
| **Exact TSP Baseline** | Held-Karp Bitmask DP | $\Theta(k^2 \cdot 2^k)$ | $\Theta(k \cdot 2^k)$ | Exponential explosion; infeasible beyond $k > 18$. |

### 4.2 Empirical Runtime Observations across Dataset Tiers
1. **Sorting Scalability ($N = 25 \to 1,500$):** Merge Sort scales smoothly from $0.03\text{ ms}$ ($N=25$) to $2.05\text{ ms}$ ($N=1,500$) with $14,877$ comparisons, whereas quadratic Insertion Sort degrades to $68.1\text{ ms}$ ($565,571$ comparisons — a **$33\times$ slowdown**).
2. **Knapsack DP Scalability:** As $(N, W)$ increases from $(20, 30)$ to $(1000, 250)$, 2D DP scales linearly with the state product $N \cdot W$ ($0.04\text{ ms} \to 12.8\text{ ms}$), while our **1D Bitmask-Optimized DP** reduces runtime to $10.4\text{ ms}$ and cuts memory usage from $O(N \cdot W)$ to $O(W)$. Across all tested scales, 0/1 DP consistently achieves higher total value and utilization than the Greedy heuristic.
3. **TSP Routing Scalability ($k = 5 \to 40$ stops):** Nearest-Neighbor TSP and 2-Opt remain under $0.50\text{ ms}$ even at $k = 40$ stops, whereas Held-Karp Exact TSP increases from $0.08\text{ ms}$ ($k=5$) to $14.3\text{ ms}$ ($k=11$) and becomes intractable for $k \ge 20$.

![Phase 4 Runtime Scalability Charts](images/phase4_runtime_scalability_charts.png)

### 4.3 Concrete Suggestions for Future Optimization
1. **1D Row-Compressed Bitmask Knapsack DP (Implemented in `src/knapsack_dp.py`):** Compresses the $(n+1)\times(W+1)$ integer matrix into a single row `dp[w]` of length $W+1$ paired with an integer bitmask `chosen_mask[w]`, reducing memory footprint by $N\times$ and improving L1 CPU cache locality.
2. **Goal-Directed A\* Search with Euclidean Landmarks:** Replacing uninformed Dijkstra with $A^*$ using admissible heuristic $h(u, t) = \text{dist}_{\text{euclid}}(u, t)$ reduces priority-queue pops by 50–70% on large metropolitan road networks.
3. **2-Opt / Or-Opt Post-Processing (Implemented in `src/graph_routing.py`):** Eliminates crossing tour edges left by greedy Nearest-Neighbor choices in $O(k^2)$ time.
4. **Joint Capacitated Vehicle Routing Problem with Time Windows (CVRPTW):** Extending single-trip Knapsack + TSP into multi-vehicle column generation or Simulated Annealing so that deferred parcels are automatically assigned to secondary vans while respecting hard SLA deadlines.

---

## 5. Phase 5: Final Report and Demonstration (20 Marks)

### 5.1 Integrated Execution Trace & Verification
Running `python3 -m src.main` produces the complete console trace below, verifying that all four algorithmic components work seamlessly together:

![Phase 5 Console Execution Screenshot](images/phase5_console_execution_screenshot.png)

### 5.2 Conclusion
The **Mini-Courier Planner** demonstrates how combining complementary algorithmic paradigms—**Merge Sort** for deterministic multi-criteria prioritization, **0/1 Knapsack Dynamic Programming** for exact payload capacity optimization, **Dijkstra's Algorithm** for road network shortest paths, and **Nearest-Neighbor + 2-Opt TSP** for scalable tour construction—solves a complex real-world logistics problem in **under 1 millisecond** for standard dispatch manifests and **under 6 milliseconds** for 500-parcel hub workloads.
