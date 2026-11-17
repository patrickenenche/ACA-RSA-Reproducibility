'''
# Cap-constrained CAFU for adaptive fragmented-spectrum aggregation
Path-level CAFU ranks fallback candidate paths by the fraction
of the current request demand that can be supplied using at most
MAX_SEGMENTS_PER_PATH accessible common-free segments:

    CAFU_p = min(C_p^cap, g_i) / g_i = LBS_p + FCS_p.

No manually selected CAFU weighting coefficients are used. Equal
CAFU scores are resolved by preferring fewer required segments
and then shorter hop count.

Fallback paths are processed sequentially in CAFU order, up to
DEFAULT_K_MAX. Processing stops when the actual allocated fragment-unit
target is satisfied. ACA-RSA and adaptive MP-SS-RSA deliberately share
this allocator in the principal r=0 experiment.
'''

from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import csv
import hashlib
import heapq
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List, Optional, Tuple


# ============================================================
# Full comparative simulator framework for EON RSA / ACA-RSA
# ------------------------------------------------------------
# Schemes included:
#   - k1_rsa                  (static K=1)
#   - k3_rsa                  (static K=3)
#   - fsa_rsa_ssfm            (dynamic SSFM single-path)
#   - fsa_k3_rsa              (dynamic SSFM top-3)
#   - rb_scsp_arsa            (repair-enhanced heuristic baseline)
#   - mp_ss_rsa               (fixed-partition multipath split-spectrum)
#   - adaptive_mp_ss_rsa      (adaptive non-coding fragment aggregation)
#   - aca_rsa                 (dynamic SSFM + adaptive fragment fallback)
#   - aca_k3_rsa              (dynamic SSFM top-3 + adaptive fragment fallback)
#   - p_aca_rsa               (proactive adaptive fragmented allocation)
#
# Metrics logged:
#   - RBP
#   - BBP
#   - accepted_load
#   - FSU
#   - avg_LFM
#   - avg_EBFM
#   - avg_K
#   - avg_r_over_g
#   - allocated and occupied slot-hop cost
#   - accepted Gbps per allocated slot-hop
#   - mean active carried Gbps per occupied slot-hop
#   - per-request decision time and per-seed simulation runtime
#   - actual aggregate offered load and traffic-trace SHA-256
#
# Modeling notes:
#   - RSA-level simulator (not physical waveform simulation)
#   - RSA-layer abstraction: one allocated slot represents one service unit
#   - no finite-field/rank, QoT, reach, guard-band, transponder, delay, or
#     control-plane model is implemented
#   - aca_rsa: classical single-path attempt first, adaptive fallback second
#   - p_aca_rsa: adaptive fragmented allocation is attempted first
# ============================================================

# ============================================================
# Configuration
# ============================================================

#TOTAL_SLOTS = 320
SLOT_GRANULARITY_GHZ = 12.5
SPECTRUM_BANDWIDTH_GHZ = 4000
TOTAL_SLOTS = int(SPECTRUM_BANDWIDTH_GHZ / SLOT_GRANULARITY_GHZ)

BPSK_M = 1
DEFAULT_K_MAX = 3
# Repair-oriented RB-SCSP proxy: one initial route plus at most three repairs.
# The cited RB-SCSP paper repeats until no route remains and only assumes K=3
# in its complexity example, so this finite cap is an explicit implementation
# choice rather than a claimed exact reproduction of the paper.
RB_SCSP_MAX_REPAIRS = 3

#ACTUAL EXPERIMENT
LOAD_VALUES = [5, 10, 15, 20, 25]
N_REQUESTS = 100000
WARMUP_REQUESTS = 0
SEEDS = list(range(1, 11))  # Paper-scale: 10 iterations / seeds

# # fOR TESTING ONLY
# LOAD_VALUES = [10, 20, 25]
# N_REQUESTS = 3000
# WARMUP_REQUESTS = 0
# SEEDS = [1, 2, 3]

BANDWIDTH_MIN_GBPS = 50
BANDWIDTH_MAX_GBPS = 200
SERVICE_RATE = 1.0  # mu
# Calibrates how the paper's offered load is mapped to the aggregate Poisson stream.
# 1 = global load interpretation; len(nodes) = per-node interpretation.
LOAD_SCALE = 9
LOAD_SCALE_BY_TOPOLOGY = {
    "GERMAN": 9,
    "USNET": 12.43, #10.555, #if this fails go back to 10.6 
    "TI": 15,
}
#Topologies
TOPOLOGIES = ["GERMAN", "USNET", "TI"]
#TOPOLOGIES = ["GERMAN", "USNET"]
#TOPOLOGIES = ["GERMAN"]   # Done!
# TOPOLOGIES = ["USNET", "TI"]
#TOPOLOGIES = ["TI"]

# The paper states that bit-rate requirements vary randomly from 50 to
# 200 Gbps, but it does not state whether the distribution is continuous
# or discrete.  Slot-uniform sampling keeps requests aligned with the
# 12.5 GHz grid and covers every possible slot width from 4 to 16 slots.
BANDWIDTH_SAMPLING_MODE = "slot_uniform"


def edge_key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)

# ------------------------------------------------------------
# ACA fallback capacity controls
# ------------------------------------------------------------

# Principal fragmentation-utilization experiment: zero redundancy.
# This experiment measures bounded adaptive fragmented-spectrum aggregation;
# it does not attribute the r=0 admission gain uniquely to coding.
MAIN_REDUNDANCY_R = 0

# Multisegment fragment-aggregation caps
MAX_SEGMENTS_PER_PATH = 3#4#2
MAX_SEGMENTS_PER_REQUEST = 6#8#4

# MP-SS-RSA controlled ablation controls: multipath split-spectrum without coding.
# It shares ACA-RSA's SSFM first stage, fallback candidate environment,
# K limit, CAFU ranking, and segment caps, but allocates predetermined
# uncoded sub-demands that must each fit contiguously.
MP_SS_K_MAX = DEFAULT_K_MAX
MP_SS_MAX_PARTS = MAX_SEGMENTS_PER_REQUEST



# Redundancy heuristic
# REDUNDANCY_ALPHA = 0.45# 0.60
# REDUNDANCY_DELTA = 1# 2

SCHEMES = [
    "k1_rsa",
    "k3_rsa",
    "fsa_rsa_ssfm",
    #"fsa_k3_rsa",
    "rb_scsp_arsa",
    "mp_ss_rsa",
    "adaptive_mp_ss_rsa",
    "aca_rsa",
    "p_aca_rsa",
    #"aca_k3_rsa",
]


EXPERIMENT_TAG = "ACA_RSA_v31_adaptive_mp_ss_metrics" # change this per experiment

# Parallelize only independent topology/scheme/load/seed runs. Each worker
# retains the original sequential event processing within its assigned run.
DEFAULT_MAX_WORKERS = min(12, max(1, os.cpu_count() or 1))

# Leave blank for a fresh run. To resume an interrupted run, paste the previous
# results folder path here so the simulator can read the existing checkpoint.
RESUME_RESULTS_DIR = os.environ.get("ACA_RSA_RESUME_RESULTS_DIR", "")

# Output paths are initialized by the parent process in main(). Windows worker
# processes import this module, so import-time directory creation is avoided.
RESULTS_DIR: Path
RAW_RESULTS_CSV: Path
SUMMARY_RESULTS_CSV: Path
CHECKPOINT_RAW_CSV: Path
CHECKPOINT_SUMMARY_CSV: Path
CHECKPOINT_CONFIG_JSON: Path
CHECKPOINT_COMPLETED_RUNS_CSV: Path


# ============================================================
# Debug / tracing
# ============================================================

ENABLE_K_DEBUG_LOG = True
K_DEBUG_CSV: Path
CHECKPOINT_K_DEBUG_CSV: Path

# Keep the per-seed K-debug checkpoint, but do not load the full debug CSV
# during finalization. Large K-debug files can exceed memory after long runs.
SAVE_FINAL_K_DEBUG_CSV = False
SAVE_PER_TOPOLOGY_K_DEBUG_CSV = False

INPUT_DEMAND_CSV: Path
CHECKPOINT_INPUT_DEMAND_CSV: Path

# Disabled for the final long run to reduce memory and disk pressure.
ENABLE_INPUT_DEMAND_LOG = False

# Resume support. A run is skipped only after its completion marker confirms
# that every required checkpoint component was flushed to disk.
RESUME_FROM_CHECKPOINT = True


def initialize_output_paths() -> None:
    """Initialize result/checkpoint paths once in the parent process."""
    global RESULTS_DIR
    global RAW_RESULTS_CSV, SUMMARY_RESULTS_CSV
    global CHECKPOINT_RAW_CSV, CHECKPOINT_SUMMARY_CSV
    global CHECKPOINT_CONFIG_JSON, CHECKPOINT_COMPLETED_RUNS_CSV
    global K_DEBUG_CSV, CHECKPOINT_K_DEBUG_CSV
    global INPUT_DEMAND_CSV, CHECKPOINT_INPUT_DEMAND_CSV

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR = (
        Path(RESUME_RESULTS_DIR)
        if RESUME_RESULTS_DIR
        else Path(f"results_{EXPERIMENT_TAG}_{timestamp}")
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    RAW_RESULTS_CSV = RESULTS_DIR / f"simulation_results_raw_{EXPERIMENT_TAG}.csv"
    SUMMARY_RESULTS_CSV = RESULTS_DIR / f"simulation_results_summary_{EXPERIMENT_TAG}.csv"
    CHECKPOINT_RAW_CSV = RESULTS_DIR / f"checkpoint_raw_per_seed_{EXPERIMENT_TAG}.csv"
    CHECKPOINT_SUMMARY_CSV = RESULTS_DIR / f"checkpoint_summary_so_far_{EXPERIMENT_TAG}.csv"
    CHECKPOINT_CONFIG_JSON = RESULTS_DIR / f"checkpoint_configuration_{EXPERIMENT_TAG}.json"
    CHECKPOINT_COMPLETED_RUNS_CSV = (
        RESULTS_DIR / f"checkpoint_completed_runs_{EXPERIMENT_TAG}.csv"
    )
    K_DEBUG_CSV = RESULTS_DIR / f"k_debug_{EXPERIMENT_TAG}.csv"
    CHECKPOINT_K_DEBUG_CSV = (
        RESULTS_DIR / f"checkpoint_k_debug_per_seed_{EXPERIMENT_TAG}.csv"
    )
    INPUT_DEMAND_CSV = RESULTS_DIR / f"input_demand_{EXPERIMENT_TAG}.csv"
    CHECKPOINT_INPUT_DEMAND_CSV = (
        RESULTS_DIR / f"checkpoint_input_demand_per_seed_{EXPERIMENT_TAG}.csv"
    )

# ============================================================
# Topology
# ============================================================

# ============================================================
# Topology
# ============================================================

def build_usnet_topology() -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    24-node, 43-link undirected USNET topology from the baseline authors'
    original adjacency-distance matrix.
    Node labels are integers 1..24.
    """
    nodes = list(range(1, 25))
 
    edges = [
        (1, 2), (1, 3),
        (2, 3), (2, 4),
        (3, 5), (3, 9), (3, 11),
        (4, 5), (4, 6), (4, 7),
        (5, 6), (5, 8), (5, 9),
        (6, 7),
        (7, 8),
        (8, 23),
        (9, 10), (9, 11), (9, 23),
        (10, 11), (10, 13), (10, 22),
        (11, 12), (11, 14),
        (12, 13), (12, 15),
        (13, 16), (13, 17), (13, 19),
        (14, 15),
        (15, 16),
        (16, 17),
        (17, 18), (17, 19),
        (18, 19), (18, 20),
        (19, 21), (19, 22),
        (20, 21),
        (21, 24),
        (22, 23), (22, 24),
        (23, 24),
    ]
    assert len(nodes) == 24
    assert len(edges) == 43
    return nodes, edges


def build_german_topology() -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    German topology from the baseline authors' original adjacency-distance
    matrix: 17 nodes, 26 bidirectional links.
    Node labels are integers 1..17.
    """
    nodes = list(range(1, 18))
    
    edges = [
        (1, 2), (1, 3),
        (2, 4), (2, 6),
        (3, 5), (3, 6), (3, 9),
        (4, 6), (4, 8),
        (5, 7),
        (6, 8), (6, 10), (6, 11),
        (7, 9),
        (8, 10),
        (9, 11),
        (10, 11), (10, 12),
        (11, 12), (11, 13),
        (12, 14), (12, 17),
        (13, 15),
        (14, 16),
        (15, 17),
        (16, 17),
    ]
    assert len(nodes) == 17
    assert len(edges) == 26
    return nodes, edges


def build_ti_topology() -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Telecom Italia (TI) topology from the baseline authors' original
    adjacency-distance matrix: 30 nodes, 50 bidirectional links.
    """
    nodes = list(range(1, 31))

    edges = [
        (1, 10), (1, 12), (1, 14), (1, 28),
        (2, 19), (2, 20), (2, 22),
        (3, 16), (3, 20), (3, 27),
        (4, 9), (4, 14), (4, 17), (4, 22),
        (5, 25), (5, 29),
        (6, 21), (6, 24),
        (7, 8), (7, 11), (7, 18),
        (8, 11), (8, 27),
        (9, 19), (9, 21), (9, 23),
        (10, 14), (10, 21), (10, 26),
        (11, 15), (11, 16), (11, 18),
        (12, 13), (12, 28),
        (13, 14), (13, 25),
        (14, 21), (14, 25),
        (15, 16), (15, 24),
        (16, 23),
        (17, 25), (17, 30),
        (19, 23),
        (20, 23),
        (21, 24),
        (22, 30),
        (23, 24),
        (26, 28),
        (29, 30),
    ]
    assert len(nodes) == 30
    assert len(edges) == 50
    return nodes, edges


def build_topology(topology_name: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Topology dispatcher.
    Supported names:
      - USNET
      - GERMAN / GER
      - TI / TELECOM_ITALIA
    """
    name = topology_name.strip().upper()

    if name == "USNET":
        return build_usnet_topology()

    if name in {"GERMAN", "GER"}:
        return build_german_topology()

    if name in {"TI", "TELECOM_ITALIA", "TELECOM-ITALIA"}:
        return build_ti_topology()

    raise ValueError(
        f"Unknown topology_name={topology_name}. "
        "Use 'USNET', 'GERMAN', or 'TI'."
    )


def get_topology_link_lengths(topology_name: str) -> Dict[Tuple[int, int], float]:
    """
    Return verified physical link lengths from Fig. 3 where they have been
    transcribed. Static K=1/K=3 RSA uses these for offline route tables.

    FSA-RSA still uses dynamic fragmentation weights (SSFM + 1), as in the
    paper. If a topology has no verified lengths here, static K paths fall back
    to unit hop weights.
    """
    name = topology_name.strip().upper()
    if name == "USNET":
        # Author-provided original USNET adjacency-distance matrix.
        lengths = {
            (1, 2): 800, (1, 3): 1000,
            (2, 3): 950, (2, 4): 1000,
            (3, 5): 1000, (3, 9): 1200, (3, 11): 1900,
            (4, 5): 1000, (4, 6): 250, (4, 7): 800,
            (5, 6): 850, (5, 8): 1150, (5, 9): 1000,
            (6, 7): 800,
            (7, 8): 1200,
            (8, 23): 900,
            (9, 10): 1000, (9, 11): 1400, (9, 23): 1000,
            (10, 11): 900, (10, 13): 1000, (10, 22): 900,
            (11, 12): 1300, (11, 14): 2600,
            (12, 13): 600, (12, 15): 1300,
            (13, 16): 1000, (13, 17): 800, (13, 19): 1000,
            (14, 15): 1200,
            (15, 16): 700,
            (16, 17): 300,
            (17, 18): 600, (17, 19): 850,
            (18, 19): 1000, (18, 20): 900,
            (19, 21): 800, (19, 22): 1100,
            (20, 21): 900,
            (21, 24): 1200,
            (22, 23): 950, (22, 24): 650, 
            (23, 24): 850,
        }
        return {edge_key(u, v): float(length) for (u, v), length in lengths.items()}

    if name in {"GERMAN", "GER"}:
        # Author-provided original German adjacency-distance matrix.
        lengths = {
            (1, 2): 144, (1, 3): 278,
            (2, 4): 114, (2, 6): 120,
            (3, 5): 37, (3, 6): 208, (3, 9): 88,
            (4, 6): 157, (4, 8): 306,
            (5, 7): 36,
            (6, 8): 298, (6, 10): 258, (6, 11): 316,
            (7, 9): 41,
            (8, 10): 174,
            (9, 11): 182,
            (10, 11): 353, (10, 12): 275,
            (11, 12): 224, (11, 13): 85,
            (12, 14): 179, (12, 17): 187,
            (13, 15): 64,
            (14, 16): 143,
            (15, 17): 74,
            (16, 17): 86,
        }
        return {edge_key(u, v): float(length) for (u, v), length in lengths.items()}

    if name in {"TI", "TELECOM_ITALIA", "TELECOM-ITALIA"}:
        # Author-provided original TI adjacency-distance matrix.
        lengths = {
            (1, 10): 113, (1, 12): 160, (1, 14): 306, (1, 28): 118,
            (2, 19): 178, (2, 20): 162, (2, 22): 107,
            (3, 16): 489, (3, 20): 289, (3, 27): 105,
            (4, 9): 101, (4, 14): 41, (4, 17): 127, (4, 22): 181,
            (5, 25): 168, (5, 29): 424,
            (6, 21): 1052, (6, 24): 674,
            (7, 8): 306, (7, 11): 265, (7, 18): 238,
            (8, 11): 45, (8, 27): 331,
            (9, 19): 197, (9, 21): 109, (9, 23): 310,
            (10, 14): 370, (10, 21): 121, (10, 26): 32,
            (11, 15): 427, (11, 16): 432, (11, 18): 438,
            (12, 13): 33, (12, 28): 210,
            (13, 14): 239, (13, 25): 183,
            (14, 21): 155, (14, 25): 123,
            (15, 16): 37, (15, 24): 252,
            (16, 23): 247,
            (17, 25): 86, (17, 30): 38,
            (19, 23): 246,
            (20, 23): 255,
            (21, 24): 472,
            (22, 30): 357,
            (23, 24): 24,
            (26, 28): 179,
            (29, 30): 196,
        }
        return {edge_key(u, v): float(length) for (u, v), length in lengths.items()}

    return {}


# ============================================================
# Event / Request models
# ============================================================

ARRIVAL = "arrival"
DEPARTURE = "departure"


@dataclass(order=True)
class Event:
    time: float
    priority: int
    event_type: str = field(compare=False)
    request_id: int = field(compare=False)


@dataclass
class Request:
    request_id: int
    src: int
    dst: int
    bandwidth_gbps: float
    slots_required: int
    arrival_time: float
    holding_time: float
    accepted: bool = False
    scheme: str = ""

    # Classical allocation
    path: Optional[List[int]] = None
    slot_start: Optional[int] = None
    slot_width: Optional[int] = None

    # Coding-aware allocation
    coded_path_allocations: List[Tuple[List[int], int, int]] = field(default_factory=list)
    split_path_allocations: List[Tuple[List[int], int, int]] = field(default_factory=list)
    K: int = 0
    g: int = 0
    r: int = 0
    used_coding: bool = False


@dataclass
class FragmentAllocationResult:
    """Resource-allocation outcome shared by ACA and adaptive MP-SS."""

    success: bool
    allocations: List[Tuple[List[int], int, int]] = field(default_factory=list)
    candidate_count: int = 0
    K_selected: int = 0
    K_effective: int = 0
    total_segments_used: int = 0
    per_path_segment_counts: List[int] = field(default_factory=list)
    per_path_link_counts: List[int] = field(default_factory=list)
    failure_reason: str = ""


# ============================================================
# Traffic generator
# ============================================================


class TrafficGenerator:
    def __init__(
        self,
        nodes: List[int],
        load_erlangs: float,
        rng: random.Random,
        topology_name: str = "",
    ):
        self.nodes = nodes
        self.load_erlangs = load_erlangs
        self.rng = rng
        self.topology_name = topology_name.upper()
        self.load_scale = LOAD_SCALE_BY_TOPOLOGY.get(self.topology_name, LOAD_SCALE)
        self.arrival_rate = load_erlangs * self.load_scale * SERVICE_RATE

        self.next_request_id = 0

    def sample_interarrival(self) -> float:
        return self.rng.expovariate(self.arrival_rate)

    def sample_holding_time(self) -> float:
        return self.rng.expovariate(SERVICE_RATE)

    def sample_bandwidth_gbps(self) -> float:
        if BANDWIDTH_SAMPLING_MODE == "slot_uniform":
            min_slots = math.ceil(BANDWIDTH_MIN_GBPS / SLOT_GRANULARITY_GHZ)
            max_slots = math.ceil(BANDWIDTH_MAX_GBPS / SLOT_GRANULARITY_GHZ)
            slots = self.rng.randint(min_slots, max_slots)
            return slots * SLOT_GRANULARITY_GHZ

        if BANDWIDTH_SAMPLING_MODE == "paper_four_rate":
            return self.rng.choice([50, 100, 150, 200])

        if BANDWIDTH_SAMPLING_MODE == "continuous_uniform":
            return self.rng.uniform(BANDWIDTH_MIN_GBPS, BANDWIDTH_MAX_GBPS)

        raise ValueError(f"Unknown BANDWIDTH_SAMPLING_MODE={BANDWIDTH_SAMPLING_MODE!r}")

    def sample_src_dst(self) -> Tuple[int, int]:
        src = self.rng.choice(self.nodes)
        dst = self.rng.choice(self.nodes)
        while dst == src:
            dst = self.rng.choice(self.nodes)
        return src, dst

    def create_request(self, now: float) -> Request:
        src, dst = self.sample_src_dst()
        bandwidth = self.sample_bandwidth_gbps()
        holding = self.sample_holding_time()
        slots_required = math.ceil(bandwidth / (SLOT_GRANULARITY_GHZ * BPSK_M))
        req = Request(
            request_id=self.next_request_id,
            src=src,
            dst=dst,
            bandwidth_gbps=bandwidth,
            slots_required=slots_required,
            arrival_time=now,
            holding_time=holding,
        )
        self.next_request_id += 1
        return req


# ============================================================
# Network / routing
# ============================================================


class Network:
    def __init__(self, nodes: List[int], undirected_edges: List[Tuple[int, int]]):
        self.nodes = nodes
        self.undirected_edges = undirected_edges
        self.adj: Dict[int, List[int]] = {n: [] for n in nodes}
        for u, v in undirected_edges:
            self.adj[u].append(v)
            self.adj[v].append(u)

    def neighbors(self, u: int) -> List[int]:
        return self.adj[u]


class RoutingEngine:
    def __init__(self, network: Network):
        self.network = network

    def dijkstra(self, src: int, dst: int, edge_weights: Dict[Tuple[int, int], float]) -> Optional[List[int]]:
        dist = {node: float("inf") for node in self.network.nodes}
        prev = {node: None for node in self.network.nodes}
        dist[src] = 0.0
        pq = [(0.0, src)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == dst:
                break
            for v in self.network.neighbors(u):
                w = edge_weights.get((u, v), float("inf"))
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        if dist[dst] == float("inf"):
            return None

        path = []
        cur = dst
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    def k_shortest_paths(self, src: int, dst: int, edge_weights: Dict[Tuple[int, int], float], k: int) -> List[List[int]]:
        first = self.dijkstra(src, dst, edge_weights)
        if first is None:
            return []

        A = [first]
        B: List[Tuple[float, List[int]]] = []

        for _ in range(1, k):
            last_path = A[-1]
            for i in range(len(last_path) - 1):
                spur_node = last_path[i]
                root_path = last_path[: i + 1]
                temp_weights = dict(edge_weights)

                for p in A:
                    if len(p) > i and p[: i + 1] == root_path:
                        u = p[i]
                        v = p[i + 1]
                        temp_weights[(u, v)] = float("inf")
                        temp_weights[(v, u)] = float("inf")

                root_path_nodes = set(root_path[:-1])
                for node in root_path_nodes:
                    for nbr in self.network.neighbors(node):
                        temp_weights[(node, nbr)] = float("inf")
                        temp_weights[(nbr, node)] = float("inf")

                spur_path = self.dijkstra(spur_node, dst, temp_weights)
                if spur_path is not None:
                    total_path = root_path[:-1] + spur_path
                    total_cost = self.path_cost(total_path, edge_weights)
                    if total_path not in A and all(candidate[1] != total_path for candidate in B):
                        heapq.heappush(B, (total_cost, total_path))

            if not B:
                break
            _, next_path = heapq.heappop(B)
            A.append(next_path)

        return A

    @staticmethod
    def path_cost(path: List[int], edge_weights: Dict[Tuple[int, int], float]) -> float:
        cost = 0.0
        for i in range(len(path) - 1):
            cost += edge_weights[(path[i], path[i + 1])]
        return cost


def precompute_static_k_paths(
    network: Network,
    routing: RoutingEngine,
    k: int = 3,
    link_lengths: Optional[Dict[Tuple[int, int], float]] = None,
) -> Dict[Tuple[int, int], List[List[int]]]:
    """
    Precompute static K shortest paths for every ordered (src, dst) pair
    using verified physical link lengths when available. These paths remain
    fixed for the whole simulation.
    """
    static_weights: Dict[Tuple[int, int], float] = {}
    for u, v in network.undirected_edges:
        weight = 1.0
        if link_lengths:
            weight = link_lengths.get(edge_key(u, v), 1.0)
        static_weights[(u, v)] = weight
        static_weights[(v, u)] = weight

    static_paths: Dict[Tuple[int, int], List[List[int]]] = {}

    for src in network.nodes:
        for dst in network.nodes:
            if src == dst:
                continue
            static_paths[(src, dst)] = routing.k_shortest_paths(src, dst, static_weights, k)

    return static_paths


# ============================================================
# Spectrum state and allocation
# ============================================================


class SpectrumEngine:
    def __init__(self, network: Network, total_slots: int):
        self.network = network
        self.total_slots = total_slots
        self.slots: Dict[Tuple[int, int], List[int]] = {}
        for u, v in self.network.undirected_edges:
            # The paper models bidirectional fiber links.  Keep one
            # physical spectrum bitmap per undirected link and expose it
            # through both ordered directions so allocation, release, FSU,
            # and fragmentation metrics all see the same resource state.
            link_slots = [0] * total_slots
            self.slots[(u, v)] = link_slots
            self.slots[(v, u)] = link_slots

    def path_links(self, path: List[int]) -> List[Tuple[int, int]]:
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    def common_free_bitmap(self, path: List[int]) -> List[int]:
        links = self.path_links(path)
        bitmap = [1] * self.total_slots
        for link in links:
            link_slots = self.slots[link]
            for i in range(self.total_slots):
                bitmap[i] &= (1 - link_slots[i])
        return bitmap

    def contiguous_blocks(self, bitmap: List[int]) -> List[Tuple[int, int]]:
        blocks = []
        start = None
        for i, val in enumerate(bitmap):
            if val == 1 and start is None:
                start = i
            elif val == 0 and start is not None:
                blocks.append((start, i - start))
                start = None
        if start is not None:
            blocks.append((start, self.total_slots - start))
        return blocks

    def largest_common_block(self, path: List[int]) -> int:
        bitmap = self.common_free_bitmap(path)
        blocks = self.contiguous_blocks(bitmap)
        return max((width for _, width in blocks), default=0)

    def total_common_free_slots(self, path: List[int]) -> int:
        return sum(self.common_free_bitmap(path))

    def first_fit(self, path: List[int], width: int) -> Optional[int]:
        bitmap = self.common_free_bitmap(path)
        run = 0
        start = 0
        for i, val in enumerate(bitmap):
            if val == 1:
                if run == 0:
                    start = i
                run += 1
                if run >= width:
                    return start
            else:
                run = 0
        return None

    def common_free_blocks_largest_first(self, path: List[int]) -> List[Tuple[int, int]]:
        """Return common-free blocks by decreasing width, then increasing slot index."""
        bitmap = self.common_free_bitmap(path)
        return sorted(self.contiguous_blocks(bitmap), key=lambda block: (-block[1], block[0]))

    def allocate_block(self, path: List[int], start: int, width: int) -> bool:
        links = self.path_links(path)
        for link in links:
            for s in range(start, start + width):
                if self.slots[link][s] == 1:
                    return False
        for link in links:
            for s in range(start, start + width):
                self.slots[link][s] = 1
        return True

    def release_block(self, path: List[int], start: int, width: int) -> None:
        for link in self.path_links(path):
            for s in range(start, start + width):
                self.slots[link][s] = 0

    def allocate_units_on_path(
        self,
        path: List[int],
        needed_units: int,
        max_segments: int = MAX_SEGMENTS_PER_PATH,
    ) -> List[Tuple[int, int]]:
        """
        Allocate UP TO `needed_units` coded units on a single path using multiple
        aligned segments, capped by `max_segments`.

        Important behavior:
        - continuity is preserved within each segment because allocation is taken
        from common_free_bitmap(path)
        - contiguity is preserved within each segment
        - different segments on the same path may use different slot indices,
        e.g. [3,4] + [10,11]
        - unlike the old version, this function does NOT roll back simply because
        the path could not satisfy the full `needed_units`
        - it returns whatever valid contribution the path can provide
        - it returns [] only if the path contributes zero units
        """

        if needed_units <= 0 or max_segments <= 0:
            return []

        blocks = self.common_free_blocks_largest_first(path)

        if not blocks:
            return []

        remaining = needed_units
        allocations: List[Tuple[int, int]] = []
        used_segments = 0

        for start, width in blocks:
            if remaining <= 0:
                break
            if used_segments >= max_segments:
                break

            take = min(width, remaining)
            if take <= 0:
                continue

            if self.allocate_block(path, start, take):
                allocations.append((start, take))
                remaining -= take
                used_segments += 1

        # Return partial contribution if any was allocated
        return allocations

    def release_coded_allocations(self, path_allocs: List[Tuple[List[int], int, int]]) -> None:
        for path, start, width in path_allocs:
            self.release_block(path, start, width)

    def occupied_slots_total(self) -> int:
        total = 0
        for u, v in self.network.undirected_edges:
            total += sum(self.slots[(u, v)])
        return total

    def link_fragment_count(self, link: Tuple[int, int]) -> int:
        bitmap = [1 - x for x in self.slots[link]]
        return len(self.contiguous_blocks(bitmap))

    def link_fragment_widths(self, link: Tuple[int, int]) -> List[int]:
        bitmap = [1 - x for x in self.slots[link]]
        return [w for _, w in self.contiguous_blocks(bitmap)]


# ============================================================
# Fragmentation and weights
# ============================================================


class FragmentationAnalyzer:
    def __init__(self, spectrum: SpectrumEngine, network: Network):
        self.spectrum = spectrum
        self.network = network

    def largest_free_block_on_link(self, link: Tuple[int, int]) -> int:
        return max(self.spectrum.link_fragment_widths(link), default=0)

    def total_free_slots_on_link(self, link: Tuple[int, int]) -> int:
        return self.spectrum.total_slots - sum(self.spectrum.slots[link])

    def fragmentation_ratio_on_link(self, link: Tuple[int, int]) -> float:
        F = self.total_free_slots_on_link(link)
        if F == 0:
            return 1.0
        L = self.largest_free_block_on_link(link)
        return 1.0 - (L / F)

    def global_lfm(self) -> float:
        total_fragments = 0
        total_slots = 0
        for u, v in self.network.undirected_edges:
            total_fragments += self.spectrum.link_fragment_count((u, v))
            total_slots += self.spectrum.total_slots
        return total_fragments / total_slots if total_slots else 0.0

    def global_ebfm(self) -> float:
        acc = 0.0
        E = len(self.network.undirected_edges)
        if E == 0:
            return 0.0
        for u, v in self.network.undirected_edges:
            widths = self.spectrum.link_fragment_widths((u, v))
            link_total = self.spectrum.total_slots
            inner = 0.0
            for w in widths:
                if w > 0:
                    inner += (w / link_total) * math.log(link_total / w)
            acc += inner
        return acc / E

class WeightEngine:
    def __init__(self, fragmentation: FragmentationAnalyzer, spectrum: SpectrumEngine, network: Network):
        self.fragmentation = fragmentation
        self.spectrum = spectrum
        self.network = network

    def unit_weights(self) -> Dict[Tuple[int, int], float]:
        weights: Dict[Tuple[int, int], float] = {}
        for u, v in self.network.undirected_edges:
            for a, b in [(u, v), (v, u)]:
                weights[(a, b)] = 1.0
        return weights


    def ssfm_style_weights(self, required_slots: int) -> Dict[Tuple[int, int], float]:
        """
        Paper-level FSA-RSA (SSFM) link weighting.

        In the SSFM paper, routing uses the current fragmentation-status
        weights omega_l = SSFM_l + 1 for all links, and only AFTER the
        shortest path is selected does the algorithm check end-to-end
        spectrum continuity and contiguity.

        Therefore, unlike RB-ARSA / requested-bandwidth-aware pruning,
        links are NOT removed simply because their current largest free
        block is smaller than the request.
        """
        weights: Dict[Tuple[int, int], float] = {}
        NFS = self.spectrum.total_slots

        for u, v in self.network.undirected_edges:
            for a, b in [(u, v), (v, u)]:
                link = (a, b)
                widths = self.spectrum.link_fragment_widths(link)
                m = len(widths)
                sum_f = sum(widths)
                f_max = max(widths, default=0)

                # Paper-consistent SSFM edge cases:
                # - fully occupied link: m = 0, f_max = 0, SSFM = 1, weight = 2
                # - all slots free: one fragment of width NFS, SSFM is minimal (>0)
                if m == 0 or sum_f == 0 or f_max == 0:
                    ssfm = 1.0
                else:
                    term1 = 1.0 - (f_max / (1.0 + sum_f))
                    term2 = 1.0 - ((sum_f / NFS) / (m + 1.0))
                    ssfm = term1 * term2
                    ssfm = max(0.0, min(1.0, ssfm))

                weights[link] = 1.0 + ssfm

        return weights

    # def ssfm_style_weights(self, required_slots: int) -> Dict[Tuple[int, int], float]:
    #     weights: Dict[Tuple[int, int], float] = {}
    #     SS = self.spectrum.total_slots
    #     for u, v in self.network.undirected_edges:
    #         for a, b in [(u, v), (v, u)]:
    #             L = self.fragmentation.largest_free_block_on_link((a, b))
    #             if L < required_slots:
    #                 weights[(a, b)] = float("inf")
    #             else:
    #                 weights[(a, b)] = SS + 1 - L
    #     return weights

    def fragment_opportunity_weights(self, required_slots: int = 1) -> Dict[Tuple[int, int], float]:
        """
        Coefficient-free residual-capacity opportunity weights for fallback
        candidate-route generation.

        For request demand g, each link receives the opportunity score

            Omega_e = min(F_e, g) / g,

        where F_e is the link's total residual free-slot capacity. The routing
        cost is its reciprocal,

            omega_e = 1 / Omega_e
                    = g / min(F_e, g),

        for F_e > 0, while fully occupied links receive infinite cost.

        The score measures request-relevant residual-capacity opportunity only.
        Actual end-to-end path usefulness and bounded fragmented accessibility
        are evaluated subsequently by the cap-constrained CAFU ranking.
        """
        weights: Dict[Tuple[int, int], float] = {}
        demand = max(1, required_slots)

        for u, v in self.network.undirected_edges:
            for a, b in [(u, v), (v, u)]:
                F = self.fragmentation.total_free_slots_on_link((a, b))
                if F <= 0:
                    weights[(a, b)] = float("inf")
                    continue

                L = self.fragmentation.largest_free_block_on_link((a, b))
                nu = min(L, demand) / demand
                psi = min(max(F - L, 0), max(demand - L, 0)) / demand
                opportunity = nu + psi
                opportunity_direct = min(F, demand) / demand

                if abs(opportunity - opportunity_direct) >= 1e-12:
                    raise AssertionError(
                        "Fragment-opportunity decomposition failed: "
                        "nu + psi != min(F, g) / g"
                    )

                opportunity = opportunity_direct
                assert 0.0 < opportunity <= 1.0
                weights[(a, b)] = 1.0 / opportunity

        return weights

    def coding_fallback_weights(self, required_slots: int = 1) -> Dict[Tuple[int, int], float]:
        """Backward-compatible alias for the neutral fragment-opportunity weights."""
        return self.fragment_opportunity_weights(required_slots)


# ============================================================
# Metrics
# ============================================================


@dataclass
class MetricsAccumulator:
    arrived_requests: int = 0
    blocked_requests: int = 0
    requested_bandwidth: float = 0.0
    blocked_bandwidth: float = 0.0
    accepted_bandwidth: float = 0.0
    accepted_requests: int = 0
    allocated_slot_hops: int = 0
    sum_K: float = 0.0
    decision_times_ns: List[int] = field(default_factory=list)

    # Debug counters
    fallback_attempts: int = 0
    fallback_successes: int = 0
    fallback_k1_successes: int = 0
    fallback_k2_successes: int = 0
    fallback_k3_successes: int = 0    

    def record_arrival(self, bandwidth: float) -> None:
        self.arrived_requests += 1
        self.requested_bandwidth += bandwidth

    def record_block(self, bandwidth: float) -> None:
        self.blocked_requests += 1
        self.blocked_bandwidth += bandwidth

    def record_accept(self, bandwidth: float, slot_hops: int) -> None:
        self.accepted_requests += 1
        self.accepted_bandwidth += bandwidth
        self.allocated_slot_hops += int(slot_hops)

    def record_decision_time(self, elapsed_ns: int) -> None:
        self.decision_times_ns.append(max(0, int(elapsed_ns)))

    def record_fallback_attempt(self) -> None:
        self.fallback_attempts += 1

    def record_fallback_success(self, K_effective: int) -> None:
        self.fallback_successes += 1
        self.sum_K += K_effective
        if K_effective == 1:
            self.fallback_k1_successes += 1
        elif K_effective == 2:
            self.fallback_k2_successes += 1
        elif K_effective >= 3:
            self.fallback_k3_successes += 1

    def rbp(self) -> float:
        return self.blocked_requests / self.arrived_requests if self.arrived_requests else 0.0

    def bbp(self) -> float:
        return self.blocked_bandwidth / self.requested_bandwidth if self.requested_bandwidth else 0.0

    def throughput(self) -> float:
        return self.accepted_bandwidth

    def avg_K(self) -> float:
        return self.sum_K / self.fallback_successes if self.fallback_successes else 0.0

    def mean_decision_time_us(self) -> float:
        return mean(self.decision_times_ns) / 1_000.0 if self.decision_times_ns else 0.0

    def median_decision_time_us(self) -> float:
        return median(self.decision_times_ns) / 1_000.0 if self.decision_times_ns else 0.0

    def p95_decision_time_us(self) -> float:
        if not self.decision_times_ns:
            return 0.0
        ordered = sorted(self.decision_times_ns)
        index = max(0, math.ceil(0.95 * len(ordered)) - 1)
        return ordered[index] / 1_000.0


# ============================================================
# Comparative simulator
# ============================================================


class ComparativeSimulator:   
    def __init__(self, scheme: str, load_erlangs: float, seed: int, topology_name: str = "USNET"):
        if scheme not in SCHEMES:
            raise ValueError(f"Unknown scheme: {scheme}")

        nodes, edges = build_topology(topology_name)
        link_lengths = get_topology_link_lengths(topology_name)
        self.rng = random.Random(seed)
        self.network = Network(nodes, edges)
        self.routing = RoutingEngine(self.network)
        self.static_k_paths = precompute_static_k_paths(
            self.network, self.routing, k=3, link_lengths=link_lengths
        )
        self.spectrum = SpectrumEngine(self.network, TOTAL_SLOTS)
        self.fragmentation = FragmentationAnalyzer(self.spectrum, self.network)
        self.weights = WeightEngine(self.fragmentation, self.spectrum, self.network)
        self.traffic = TrafficGenerator(nodes, load_erlangs, self.rng, topology_name)

        self.scheme = scheme
        self.topology_name = topology_name
        self.load_erlangs = load_erlangs
        self.seed = seed

        self.time = 0.0
        self.event_queue: List[Event] = []
        self.active_requests: Dict[int, Request] = {}
        self.metrics = MetricsAccumulator()
        self.snapshot_occupied_sum = 0.0
        self.snapshot_occupied_slot_hops_sum = 0.0
        self.snapshot_carried_gbps_sum = 0.0
        self.snapshot_lfm_sum = 0.0
        self.snapshot_ebfm_sum = 0.0
        self.snapshot_count = 0
        self.simulation_runtime_seconds = 0.0

        # The trace fingerprint excludes the scheme so matched runs can verify
        # that they received the exact same offered request sequence.
        self.traffic_trace_hasher = hashlib.sha256()
        trace_header = (
            f"traffic-trace-v1|{self.topology_name}|{float(self.load_erlangs).hex()}|"
            f"{self.seed}|{BANDWIDTH_SAMPLING_MODE}\n"
        )
        self.traffic_trace_hasher.update(trace_header.encode("ascii"))

        self.k_debug_rows: List[Dict[str, object]] = []

        self.input_demand_rows: List[Dict[str, object]] = []

    def _update_traffic_trace(self, req: Request) -> None:
        """Hash one generated request using a stable, scheme-independent encoding."""
        trace_row = (
            f"{req.request_id}|{req.arrival_time.hex()}|{req.src}|{req.dst}|"
            f"{req.bandwidth_gbps.hex()}|{req.slots_required}|"
            f"{req.holding_time.hex()}\n"
        )
        self.traffic_trace_hasher.update(trace_row.encode("ascii"))

    def _log_k_debug(
        self,
        req: Request,
        K_selected: int,
        K_effective: int,
        success: bool,
        g: int,
        r: int,
        total_segments_used: int,
        per_path_segment_counts: List[int],
        per_path_link_counts: List[int],
        failure_reason: str,
    ) -> None:
        if not ENABLE_K_DEBUG_LOG:
            return

        wasted_path_ratio = (
            1.0 - (K_effective / K_selected)
            if K_selected > 0
            else None
        )

        row = {
            "scheme": self.scheme,
            "topology": self.topology_name,
            "seed": self.seed,
            "load": self.load_erlangs,
            "request_id": req.request_id,
            "src": req.src,
            "dst": req.dst,
            "slots_required": req.slots_required,
            "K_selected": K_selected,
            "K_effective": K_effective,
            "success": int(success),
            "failure_reason": failure_reason,
            "g": g,
            "r": r,
            # "g": req.g,
            # "r": req.r,
            "total_segments_used": total_segments_used,
            "per_path_segment_counts": "|".join(str(x) for x in per_path_segment_counts),
            "per_path_link_counts": "|".join(str(x) for x in per_path_link_counts),
            "wasted_path_ratio": wasted_path_ratio,
            "NPR": wasted_path_ratio,
        }
        self.k_debug_rows.append(row)
    
    def _log_input_demand(self, req: Request) -> None:
        """
        Log every generated request before any routing/allocation decision.
        This captures the offered-traffic demand distribution.
        """
        if not ENABLE_INPUT_DEMAND_LOG:
            return
        self.input_demand_rows.append({
            "scheme": self.scheme,
            "topology": self.topology_name,
            "seed": self.seed,
            "load": self.load_erlangs,
            "request_id": req.request_id,
            "src": req.src,
            "dst": req.dst,
            "bandwidth_gbps": req.bandwidth_gbps,
            "slots_required": req.slots_required,
        })

    # ---------------------------------------------------------
    # Main simulation
    # ---------------------------------------------------------
    def run(self, n_requests: int, warmup_requests: int = 0) -> Dict[str, float]:
        run_started = time.perf_counter()
        now = 0.0
        first_request = self.traffic.create_request(now)
        self._update_traffic_trace(first_request)
        self.active_requests[first_request.request_id] = first_request
        heapq.heappush(self.event_queue, Event(now, 0, ARRIVAL, first_request.request_id))

        generated = 1
        processed_arrivals = 0

        while processed_arrivals < n_requests and self.event_queue:
            ev = heapq.heappop(self.event_queue)
            self.time = ev.time

            if ev.event_type == ARRIVAL:
                req = self.active_requests[ev.request_id]
                in_steady_state = processed_arrivals >= warmup_requests
                self._handle_arrival(req, in_steady_state)
                if not req.accepted:
                    self.active_requests.pop(req.request_id, None)
                if in_steady_state:
                    # Use one common observation epoch per offered request.
                    # Departures update state but do not create extra samples.
                    self._record_state_snapshot()
                processed_arrivals += 1

                if generated < n_requests:
                    next_time = self.time + self.traffic.sample_interarrival()
                    next_req = self.traffic.create_request(next_time)
                    self._update_traffic_trace(next_req)
                    self.active_requests[next_req.request_id] = next_req
                    heapq.heappush(self.event_queue, Event(next_time, 0, ARRIVAL, next_req.request_id))
                    generated += 1

            elif ev.event_type == DEPARTURE:
                req = self.active_requests.get(ev.request_id)
                if req is not None:
                    self._handle_departure(req)
                    del self.active_requests[ev.request_id]

        self.simulation_runtime_seconds = time.perf_counter() - run_started
        return self._finalize_results()

    def _record_state_snapshot(self) -> None:
        capacity = len(self.network.undirected_edges) * TOTAL_SLOTS
        occupied_slot_hops = self.spectrum.occupied_slots_total()
        active_carried_gbps = sum(
            req.bandwidth_gbps
            for req in self.active_requests.values()
            if req.accepted
        )
        self.snapshot_occupied_sum += occupied_slot_hops / capacity if capacity else 0.0
        self.snapshot_occupied_slot_hops_sum += occupied_slot_hops
        self.snapshot_carried_gbps_sum += active_carried_gbps
        self.snapshot_lfm_sum += self.fragmentation.global_lfm()
        self.snapshot_ebfm_sum += self.fragmentation.global_ebfm()
        self.snapshot_count += 1

    # ---------------------------------------------------------
    # Arrival/departure
    # ---------------------------------------------------------
    def _dispatch_scheme(self, req: Request, in_steady_state: bool) -> bool:
        """Execute only the route/spectrum decision for one request."""
        if self.scheme == "k1_rsa":
            return self._scheme_k1_rsa(req)
        if self.scheme == "k3_rsa":
            return self._scheme_k3_rsa(req)
        if self.scheme == "fsa_rsa_ssfm":
            return self._scheme_fsa_rsa_ssfm(req)
        if self.scheme == "fsa_k3_rsa":
            return self._scheme_fsa_k3_rsa(req)
        if self.scheme == "rb_scsp_arsa":
            return self._scheme_rb_scsp_arsa(req)
        if self.scheme == "mp_ss_rsa":
            return self._scheme_mp_ss_rsa(req)
        if self.scheme == "adaptive_mp_ss_rsa":
            return self._scheme_adaptive_mp_ss_rsa(req, in_steady_state)
        if self.scheme == "aca_rsa":
            return self._scheme_aca_rsa(req, in_steady_state)
        if self.scheme == "p_aca_rsa":
            return self._scheme_p_aca_rsa(req, in_steady_state)
        if self.scheme == "aca_k3_rsa":
            return self._scheme_aca_k3_rsa(req, in_steady_state)
        raise ValueError(f"Unknown scheme: {self.scheme}")

    @staticmethod
    def _request_slot_hops(req: Request) -> int:
        """Return the accepted request's allocated spectrum-slot hop cost."""
        if req.coded_path_allocations:
            return sum(width * (len(path) - 1) for path, _, width in req.coded_path_allocations)
        if req.split_path_allocations:
            return sum(width * (len(path) - 1) for path, _, width in req.split_path_allocations)
        if req.path is not None and req.slot_width is not None:
            return req.slot_width * (len(req.path) - 1)
        return 0

    def _handle_arrival(self, req: Request, in_steady_state: bool) -> None:
        if in_steady_state:
            self.metrics.record_arrival(req.bandwidth_gbps)
            if ENABLE_INPUT_DEMAND_LOG:
                self._log_input_demand(req)

        decision_started = time.perf_counter_ns()
        accepted = self._dispatch_scheme(req, in_steady_state)
        decision_elapsed_ns = time.perf_counter_ns() - decision_started
        if in_steady_state:
            self.metrics.record_decision_time(decision_elapsed_ns)

        if not accepted and in_steady_state:
            self.metrics.record_block(req.bandwidth_gbps)
        elif accepted and in_steady_state:
            self.metrics.record_accept(req.bandwidth_gbps, self._request_slot_hops(req))

        if accepted:
            depart_time = self.time + req.holding_time
            heapq.heappush(self.event_queue, Event(depart_time, 1, DEPARTURE, req.request_id))

    def _handle_departure(self, req: Request) -> None:
        if not req.accepted:
            return
        if req.scheme in {
            "k1_rsa", "k3_rsa", "fsa_rsa_ssfm", "fsa_k3_rsa", "rb_scsp_arsa",
            "mp_ss_rsa", "adaptive_mp_ss_rsa",
            "aca_rsa_classical", "aca_k3_rsa_classical", "p_aca_rsa_classical_rescue"
        }:
            if req.path is not None and req.slot_start is not None and req.slot_width is not None:
                self.spectrum.release_block(req.path, req.slot_start, req.slot_width)
        elif req.scheme in {
            "aca_rsa_coding_fallback",
            "aca_k3_rsa_coding_fallback",
            "p_aca_rsa_coding",
        }:
            self.spectrum.release_coded_allocations(req.coded_path_allocations)
        elif req.scheme in {"mp_ss_rsa_split", "adaptive_mp_ss_rsa_split"}:
            self.spectrum.release_coded_allocations(req.split_path_allocations)

    # ---------------------------------------------------------
    # Baseline schemes
    # ---------------------------------------------------------
    def _scheme_k1_rsa(self, req: Request) -> bool:
        paths = self.static_k_paths.get((req.src, req.dst), [])
        if not paths:
            return False
        path = paths[0]
        start = self.spectrum.first_fit(path, req.slots_required)
        if start is None:
            return False
        if not self.spectrum.allocate_block(path, start, req.slots_required):
            return False
        req.accepted = True
        req.scheme = "k1_rsa"
        req.path = path
        req.slot_start = start
        req.slot_width = req.slots_required
        return True

    def _scheme_k3_rsa(self, req: Request) -> bool:
        paths = self.static_k_paths.get((req.src, req.dst), [])
        if not paths:
            return False
        for path in paths[:3]:
            start = self.spectrum.first_fit(path, req.slots_required)
            if start is None:
                continue
            if self.spectrum.allocate_block(path, start, req.slots_required):
                req.accepted = True
                req.scheme = "k3_rsa"
                req.path = path
                req.slot_start = start
                req.slot_width = req.slots_required
                return True
        return False

    def _scheme_fsa_rsa_ssfm(self, req: Request) -> bool:
        weights = self.weights.ssfm_style_weights(req.slots_required)
        path = self.routing.dijkstra(req.src, req.dst, weights)
        if path is None:
            return False
        start = self.spectrum.first_fit(path, req.slots_required)
        if start is None:
            return False
        if not self.spectrum.allocate_block(path, start, req.slots_required):
            return False
        req.accepted = True
        req.scheme = "fsa_rsa_ssfm"
        req.path = path
        req.slot_start = start
        req.slot_width = req.slots_required
        return True

    def _scheme_fsa_k3_rsa(self, req: Request) -> bool:
        """
        Dynamic SSFM + K3 baseline:
        recompute SSFM weights at each arrival,
        get dynamic top-3 shortest paths,
        try classical contiguous allocation path-by-path.
        No coding.
        """
        weights = self.weights.ssfm_style_weights(req.slots_required)
        paths = self.routing.k_shortest_paths(req.src, req.dst, weights, 3)
        if not paths:
            return False

        for path in paths:
            start = self.spectrum.first_fit(path, req.slots_required)
            if start is None:
                continue
            if self.spectrum.allocate_block(path, start, req.slots_required):
                req.accepted = True
                req.scheme = "fsa_k3_rsa"
                req.path = path
                req.slot_start = start
                req.slot_width = req.slots_required
                return True

        return False

    def _scheme_rb_scsp_arsa(self, req: Request) -> bool:
        """Repair-oriented RB-SCSP proxy with an explicit three-repair cap."""
        base_weights = self.weights.ssfm_style_weights(req.slots_required)
        working_weights = dict(base_weights)

        for _ in range(RB_SCSP_MAX_REPAIRS + 1):
            path = self.routing.dijkstra(req.src, req.dst, working_weights)
            if path is None:
                return False

            start = self.spectrum.first_fit(path, req.slots_required)
            if start is not None and self.spectrum.allocate_block(path, start, req.slots_required):
                req.accepted = True
                req.scheme = "rb_scsp_arsa"
                req.path = path
                req.slot_start = start
                req.slot_width = req.slots_required
                return True

            # repair: penalize most blocking / fragmented link on current path
            weakest_link = None
            weakest_L = float("inf")
            for link in self.spectrum.path_links(path):
                L = self.fragmentation.largest_free_block_on_link(link)
                if L < weakest_L:
                    weakest_L = L
                    weakest_link = link
            if weakest_link is None:
                return False
            u, v = weakest_link
            working_weights[(u, v)] = float("inf")
            working_weights[(v, u)] = float("inf")

        return False

    def _rank_fallback_candidates(
        self,
        req: Request,
        k_max: int = DEFAULT_K_MAX,
    ) -> Tuple[List[Tuple[float, int, int, List[int], Dict[str, object], Dict[str, float]]], int]:
        """Return fallback paths ranked by the same cap-aware CAFU used by ACA-RSA."""
        edge_weights = self.weights.fragment_opportunity_weights(req.slots_required)
        candidates = self.routing.k_shortest_paths(req.src, req.dst, edge_weights, k_max)

        ranked: List[Tuple[float, int, int, List[int], Dict[str, object], Dict[str, float]]] = []
        demand = max(1, req.slots_required)

        for path in candidates:
            stats = self._compute_path_fragment_stats(path)
            Fp = stats["Fp"]
            capped_units = stats["capped_units"]

            if capped_units <= 0 or Fp <= 0:
                continue

            cafu_parts = self._cap_constrained_cafu_components(
                stats["top_segment_widths"], demand
            )
            cafu = cafu_parts["cafu"]
            segments_needed = cafu_parts["segments_needed"]
            hop_count = len(path) - 1

            ranked.append((cafu, segments_needed, hop_count, path, stats, cafu_parts))

        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        return ranked, len(candidates)

    @staticmethod
    def _mp_ss_partition_demand(g: int, max_parts: int = MP_SS_MAX_PARTS) -> List[int]:
        """Predetermined non-coding sub-demands for the controlled MP-SS ablation."""
        g = int(g)
        if g <= 0:
            return []
        J = min(max(1, int(max_parts)), g)
        q, remainder = divmod(g, J)
        return [q + 1] * remainder + [q] * (J - remainder)

    def _scheme_mp_ss_rsa(self, req: Request) -> bool:
        """
        Controlled MP-SS-RSA ablation: multipath split-spectrum without coding.

        The scheme shares ACA-RSA's SSFM first stage and fallback candidate
        environment. In fallback, however, the request is partitioned into
        predetermined uncoded sub-demands. Each sub-demand must be allocated
        completely as one contiguous block on one ranked fallback path. Unlike
        ACA-RSA, partial fragments are not interchangeable coded units.
        """

        # Stage 1: same dynamic SSFM contiguous attempt used by ACA-RSA.
        weights = self.weights.ssfm_style_weights(req.slots_required)
        path = self.routing.dijkstra(req.src, req.dst, weights)
        if path is not None:
            start = self.spectrum.first_fit(path, req.slots_required)
            if start is not None and self.spectrum.allocate_block(path, start, req.slots_required):
                req.accepted = True
                req.scheme = "mp_ss_rsa"
                req.path = path
                req.slot_start = start
                req.slot_width = req.slots_required
                req.g = req.slots_required
                req.r = 0
                req.used_coding = False
                return True

        # Stage 2: same fallback candidates, ranking, and largest-fragment-first
        # segment order as ACA-RSA, but with non-coding split-spectrum feasibility.
        # Each predetermined part is indivisible and contiguous.
        ranked, _ = self._rank_fallback_candidates(req, MP_SS_K_MAX)
        if not ranked:
            return False

        candidate_paths = [item[3] for item in ranked[:MP_SS_K_MAX]]
        split_widths = self._mp_ss_partition_demand(req.slots_required, MP_SS_MAX_PARTS)
        if not split_widths:
            return False

        remaining_parts = list(split_widths)
        allocations: List[Tuple[List[int], int, int]] = []
        total_segments = 0

        # Deterministic sequential greedy allocation. Paths follow CAFU order,
        # parts retain partition order, and blocks use width-descending/index-
        # ascending order. Earlier placements are never reconsidered.
        for split_path in candidate_paths:
            if not remaining_parts or total_segments >= MAX_SEGMENTS_PER_REQUEST:
                break

            path_segments_used = 0
            while (
                remaining_parts
                and path_segments_used < MAX_SEGMENTS_PER_PATH
                and total_segments < MAX_SEGMENTS_PER_REQUEST
            ):
                blocks = self.spectrum.common_free_blocks_largest_first(split_path)
                selected = None
                for part_index, width in enumerate(remaining_parts):
                    for start, block_width in blocks:
                        if block_width >= width:
                            selected = (part_index, width, start)
                            break
                    if selected is not None:
                        break

                if selected is None:
                    break

                part_index, width, start = selected
                if not self.spectrum.allocate_block(split_path, start, width):
                    raise RuntimeError("MP-SS allocation changed during deterministic placement.")

                allocations.append((split_path, start, width))
                remaining_parts.pop(part_index)
                path_segments_used += 1
                total_segments += 1

        if remaining_parts:
            self.spectrum.release_coded_allocations(allocations)
            allocations.clear()
            return False

        req.accepted = True
        req.scheme = "mp_ss_rsa_split"
        req.split_path_allocations = list(allocations)
        req.g = req.slots_required
        req.r = 0
        req.used_coding = False
        req.K = len({tuple(path) for path, _, _ in allocations})
        return True

    # ---------------------------------------------------------
    # Proposed scheme
    # ---------------------------------------------------------
    def _try_ssfm_contiguous(self, req: Request, scheme_name: str) -> bool:
        """Shared Stage-1 SSFM contiguous attempt for ACA and adaptive MP-SS."""
        weights = self.weights.ssfm_style_weights(req.slots_required)
        path = self.routing.dijkstra(req.src, req.dst, weights)
        if path is None:
            return False
        start = self.spectrum.first_fit(path, req.slots_required)
        if start is None or not self.spectrum.allocate_block(path, start, req.slots_required):
            return False

        req.accepted = True
        req.scheme = scheme_name
        req.path = path
        req.slot_start = start
        req.slot_width = req.slots_required
        req.g = req.slots_required
        req.r = MAIN_REDUNDANCY_R
        req.used_coding = False
        return True

    def _scheme_adaptive_mp_ss_rsa(self, req: Request, in_steady_state: bool) -> bool:
        """
        Strong adaptive non-coding comparator for the principal r=0 study.

        Stage 1 and the fallback resource allocator are identical to ACA-RSA.
        In fallback, each observed fragment contribution is interpreted as an
        adaptively sized uncoded sub-demand rather than an interchangeable coded
        unit. Source-unit identity is immaterial to r=0 admission feasibility.
        """
        if self._try_ssfm_contiguous(req, "adaptive_mp_ss_rsa"):
            return True
        return self._adaptive_fragment_fallback(
            req,
            in_steady_state,
            allocation_mode="adaptive_uncoded",
        )

    def _scheme_aca_rsa(self, req: Request, in_steady_state: bool) -> bool:
        if self._try_ssfm_contiguous(req, "aca_rsa_classical"):
            return True
        return self._coding_fallback(req, in_steady_state)

    def _scheme_p_aca_rsa(self, req: Request, in_steady_state: bool) -> bool:
        """
        Proactive ACA-RSA, Option A:
        directly activate coding-aware fragmented allocation for every request.

        Option B, kept below as a commented rescue variant for later testing:
        if proactive coding fails, try the classical SSFM contiguous allocation.
        """
        ok = self._coding_fallback(req, in_steady_state)
        if ok and req.scheme == "aca_rsa_coding_fallback":
            req.scheme = "p_aca_rsa_coding"
        return ok

        # Option B: proactive coding first, SSFM contiguous rescue second.
        # ok = self._coding_fallback(req, in_steady_state)
        # if ok and req.scheme == "aca_rsa_coding_fallback":
        #     req.scheme = "p_aca_rsa_coding"
        #     return True
        #
        # weights = self.weights.ssfm_style_weights(req.slots_required)
        # path = self.routing.dijkstra(req.src, req.dst, weights)
        # if path is not None:
        #     start = self.spectrum.first_fit(path, req.slots_required)
        #     if start is not None and self.spectrum.allocate_block(path, start, req.slots_required):
        #         req.accepted = True
        #         req.scheme = "p_aca_rsa_classical_rescue"
        #         req.path = path
        #         req.slot_start = start
        #         req.slot_width = req.slots_required
        #         return True
        # return False

    def _scheme_aca_k3_rsa(self, req: Request, in_steady_state: bool) -> bool:
        """
        Dynamic SSFM + K3 + coding:
        1. Recompute SSFM weights at each arrival
        2. Try up to 3 SSFM-ranked paths classically
        3. If all fail, fall back to coding
        """
        weights = self.weights.ssfm_style_weights(req.slots_required)
        paths = self.routing.k_shortest_paths(req.src, req.dst, weights, 3)

        for path in paths:
            start = self.spectrum.first_fit(path, req.slots_required)
            if start is None:
                continue
            if self.spectrum.allocate_block(path, start, req.slots_required):
                req.accepted = True
                req.scheme = "aca_k3_rsa_classical"
                req.path = path
                req.slot_start = start
                req.slot_width = req.slots_required
                return True

        ok = self._coding_fallback(req, in_steady_state)
        if ok and req.scheme == "aca_rsa_coding_fallback":
            req.scheme = "aca_k3_rsa_coding_fallback"
        return ok

    def _allocate_adaptive_fragments(
        self,
        req: Request,
        target_units: int,
        k_max: int = DEFAULT_K_MAX,
    ) -> FragmentAllocationResult:
        """
        Allocate a bounded aggregate of currently available fragments.

        This method contains the complete resource-selection behavior shared by
        adaptive MP-SS-RSA and ACA-RSA. It has no coding-specific admission rule
        and records no scheme-specific metrics. Allocation is immediate, so
        overlapping paths observe the same updated link-slot bitmap.
        """
        ranked, candidate_count = self._rank_fallback_candidates(req, k_max)
        if candidate_count <= 0:
            return FragmentAllocationResult(
                success=False,
                candidate_count=candidate_count,
                failure_reason="no_candidate_paths",
            )
        if not ranked:
            return FragmentAllocationResult(
                success=False,
                candidate_count=candidate_count,
                failure_reason="no_usable_ranked_paths",
            )

        allocations: List[Tuple[List[int], int, int]] = []
        remaining = max(0, int(target_units))
        total_segments_used = 0
        per_path_segment_counts: List[int] = []
        per_path_link_counts: List[int] = []
        K_selected = 0

        for _, _, _, path, stats, _ in ranked:
            if remaining <= 0 or K_selected >= k_max:
                break
            if total_segments_used >= MAX_SEGMENTS_PER_REQUEST:
                break

            usable_units = int(stats["capped_units"])
            if usable_units <= 0:
                continue

            K_selected += 1
            desired_here = min(usable_units, remaining)
            remaining_segment_budget = MAX_SEGMENTS_PER_REQUEST - total_segments_used
            per_path_segment_budget = min(MAX_SEGMENTS_PER_PATH, remaining_segment_budget)
            if desired_here <= 0 or per_path_segment_budget <= 0:
                break

            block_allocs = self.spectrum.allocate_units_on_path(
                path,
                desired_here,
                max_segments=per_path_segment_budget,
            )
            if not block_allocs:
                continue

            allocated_here = sum(width for _, width in block_allocs)
            if allocated_here <= 0:
                continue

            allocations.extend((path, start, width) for start, width in block_allocs)
            per_path_segment_counts.append(len(block_allocs))
            per_path_link_counts.append(len(self.spectrum.path_links(path)))
            remaining -= allocated_here
            total_segments_used += len(block_allocs)

        K_effective = len(per_path_segment_counts)
        if remaining > 0:
            if total_segments_used >= MAX_SEGMENTS_PER_REQUEST:
                failure_reason = "segment_budget_exhausted"
            elif K_selected >= k_max:
                failure_reason = "candidate_path_limit_reached"
            else:
                failure_reason = "insufficient_aggregate_units"
            self.spectrum.release_coded_allocations(allocations)
            return FragmentAllocationResult(
                success=False,
                candidate_count=candidate_count,
                K_selected=K_selected,
                K_effective=K_effective,
                total_segments_used=total_segments_used,
                per_path_segment_counts=per_path_segment_counts,
                per_path_link_counts=per_path_link_counts,
                failure_reason=failure_reason,
            )

        return FragmentAllocationResult(
            success=True,
            allocations=allocations,
            candidate_count=candidate_count,
            K_selected=K_selected,
            K_effective=K_effective,
            total_segments_used=total_segments_used,
            per_path_segment_counts=per_path_segment_counts,
            per_path_link_counts=per_path_link_counts,
            failure_reason="success",
        )

    def _adaptive_fragment_fallback(
        self,
        req: Request,
        in_steady_state: bool,
        allocation_mode: str,
    ) -> bool:
        """Apply the shared allocator and attach only the requested semantics."""
        if allocation_mode not in {"coded", "adaptive_uncoded"}:
            raise ValueError(f"Unknown adaptive fallback mode: {allocation_mode}")

        g = req.slots_required
        r = MAIN_REDUNDANCY_R
        req.g = g
        req.r = r

        if in_steady_state:
            self.metrics.record_fallback_attempt()

        outcome = self._allocate_adaptive_fragments(
            req,
            target_units=g + r,
            k_max=DEFAULT_K_MAX,
        )

        if in_steady_state:
            self._log_k_debug(
                req=req,
                K_selected=outcome.K_selected,
                K_effective=outcome.K_effective,
                success=outcome.success,
                g=g,
                r=r,
                total_segments_used=outcome.total_segments_used,
                per_path_segment_counts=outcome.per_path_segment_counts,
                per_path_link_counts=outcome.per_path_link_counts,
                failure_reason=outcome.failure_reason,
            )

        if not outcome.success:
            return False

        req.accepted = True
        req.K = outcome.K_effective
        if allocation_mode == "coded":
            req.scheme = "aca_rsa_coding_fallback"
            req.coded_path_allocations = list(outcome.allocations)
            req.used_coding = True
        else:
            req.scheme = "adaptive_mp_ss_rsa_split"
            req.split_path_allocations = list(outcome.allocations)
            req.used_coding = False

        if in_steady_state:
            self.metrics.record_fallback_success(outcome.K_effective)
        return True

    def _coding_fallback(self, req: Request, in_steady_state: bool) -> bool:
        """ACA wrapper around the shared adaptive fragment allocator."""
        return self._adaptive_fragment_fallback(
            req,
            in_steady_state,
            allocation_mode="coded",
        )
    

    @staticmethod
    def _cap_constrained_cafu_components(top_segment_widths: List[int], demand: int) -> Dict[str, float]:
        """Return coefficient-free CAFU decomposition and deterministic tie data."""
        demand = max(1, int(demand))
        widths = [max(0, int(width)) for width in top_segment_widths if int(width) > 0]
        capped_units = sum(widths)
        Lp = max(widths, default=0)

        cafu = min(capped_units, demand) / demand
        lbs = min(Lp, demand) / demand
        remaining_after_largest = max(demand - Lp, 0)
        additional_capped_units = max(capped_units - Lp, 0)
        fcs = min(additional_capped_units, remaining_after_largest) / demand

        if abs(cafu - (lbs + fcs)) >= 1e-12:
            raise AssertionError("CAFU decomposition failed: CAFU != LBS + FCS")

        useful_target = min(capped_units, demand)
        segments_needed = 0
        accumulated = 0
        if useful_target > 0:
            for width in widths:
                segments_needed += 1
                accumulated += width
                if accumulated >= useful_target:
                    break

        return {
            "cafu": cafu,
            "lbs": lbs,
            "fcs": fcs,
            "segments_needed": segments_needed,
            "useful_target": useful_target,
        }

    def _compute_path_fragment_stats(self, path: List[int]) -> Dict[str, object]:
        """
        Build cap-constrained path stats from the exact common-free blocks that
        are reachable by the allocator. CAFU uses only the largest
        MAX_SEGMENTS_PER_PATH common-free segments.
        """
        bitmap = self.spectrum.common_free_bitmap(path)
        blocks = self.spectrum.contiguous_blocks(bitmap)

        if not blocks:
            return {
                "blocks": [],
                "Fp": 0,
                "Lp": 0,
                "phi": 1.0,
                "top_segment_widths": [],
                "capped_units": 0,
            }

        widths = [width for _, width in blocks]
        Fp = sum(widths)
        Lp = max(widths)
        phi = 1.0 - (Lp / Fp) if Fp > 0 else 1.0
        top_segment_widths = sorted(widths, reverse=True)[:MAX_SEGMENTS_PER_PATH]
        capped_units = sum(top_segment_widths)

        return {
            "blocks": blocks,
            "Fp": Fp,
            "Lp": Lp,
            "phi": phi,
            "top_segment_widths": top_segment_widths,
            "capped_units": capped_units,
        }


    # The principal simulator uses r_i = 0 directly. Adaptive redundancy and
    # single-erasure r_i = 1 experiments are kept in separate scripts.

    def _finalize_results(self) -> Dict[str, float]:
        rbp = self.metrics.rbp()
        bbp = self.metrics.bbp()
        accepted_load = self.load_erlangs * (1.0 - rbp)
        aggregate_offered_load = self.traffic.arrival_rate / SERVICE_RATE
        aggregate_accepted_load = aggregate_offered_load * (1.0 - rbp)

        throughput_sum = self.metrics.throughput()
        throughput_rate = throughput_sum / self.time if self.time > 0 else 0.0
        throughput_norm = (
            throughput_sum / self.metrics.requested_bandwidth
            if self.metrics.requested_bandwidth > 0 else 0.0
        )

        avg_fsu = self.snapshot_occupied_sum / self.snapshot_count if self.snapshot_count else 0.0
        avg_occupied_slot_hops = (
            self.snapshot_occupied_slot_hops_sum / self.snapshot_count
            if self.snapshot_count else 0.0
        )
        mean_active_carried_gbps = (
            self.snapshot_carried_gbps_sum / self.snapshot_count
            if self.snapshot_count else 0.0
        )
        carried_gbps_per_occupied_slot_hop = (
            mean_active_carried_gbps / avg_occupied_slot_hops
            if avg_occupied_slot_hops > 0 else 0.0
        )
        occupied_slot_hops_per_carried_gbps = (
            avg_occupied_slot_hops / mean_active_carried_gbps
            if mean_active_carried_gbps > 0 else 0.0
        )
        lfm = self.snapshot_lfm_sum / self.snapshot_count if self.snapshot_count else self.fragmentation.global_lfm()
        ebfm = self.snapshot_ebfm_sum / self.snapshot_count if self.snapshot_count else self.fragmentation.global_ebfm()

        allocated_slot_hops = self.metrics.allocated_slot_hops
        mean_slot_hops_per_accepted_request = (
            allocated_slot_hops / self.metrics.accepted_requests
            if self.metrics.accepted_requests else 0.0
        )
        slot_hops_per_accepted_gbps = (
            allocated_slot_hops / self.metrics.accepted_bandwidth
            if self.metrics.accepted_bandwidth > 0 else 0.0
        )
        accepted_gbps_per_slot_hop = (
            self.metrics.accepted_bandwidth / allocated_slot_hops
            if allocated_slot_hops > 0 else 0.0
        )

        return {
            "scheme": self.scheme,
            "topology": self.topology_name,
            "load": self.load_erlangs,
            "nominal_load_index": self.load_erlangs,
            "load_scale": self.traffic.load_scale,
            "actual_aggregate_offered_load_erlangs": aggregate_offered_load,
            "actual_aggregate_accepted_load_erlangs": aggregate_accepted_load,
            "seed": self.seed,
            "RBP": rbp,
            "BBP": bbp,
            "accepted_load": accepted_load,

            "throughput": throughput_norm,
            "throughput_sum": throughput_sum,
            "throughput_rate": throughput_rate,

            "FSU": avg_fsu,
            "avg_LFM": lfm,
            "FDM": lfm,
            "avg_EBFM": ebfm,
            "avg_K": self.metrics.avg_K(),
            "avg_r_over_g": 0.0,

            "accepted_request_count": self.metrics.accepted_requests,
            "total_allocated_slot_hops": allocated_slot_hops,
            "mean_allocated_slot_hops_per_accepted_request": mean_slot_hops_per_accepted_request,
            "slot_hops_per_accepted_gbps": slot_hops_per_accepted_gbps,
            "accepted_gbps_per_slot_hop": accepted_gbps_per_slot_hop,
            "mean_occupied_slot_hops": avg_occupied_slot_hops,
            "mean_active_carried_gbps": mean_active_carried_gbps,
            "carried_gbps_per_occupied_slot_hop": carried_gbps_per_occupied_slot_hop,
            "occupied_slot_hops_per_carried_gbps": occupied_slot_hops_per_carried_gbps,

            "decision_sample_count": len(self.metrics.decision_times_ns),
            "mean_decision_time_us": self.metrics.mean_decision_time_us(),
            "median_decision_time_us": self.metrics.median_decision_time_us(),
            "p95_decision_time_us": self.metrics.p95_decision_time_us(),
            "simulation_runtime_seconds": self.simulation_runtime_seconds,
            "traffic_trace_sha256": self.traffic_trace_hasher.hexdigest(),
        }


# ============================================================
# Experiment framework
# ============================================================

def scientific_configuration() -> Dict[str, object]:
    """Return the complete result-affecting configuration used for resume checks."""
    topology_signatures: Dict[str, object] = {}
    for topology_name in TOPOLOGIES:
        nodes, edges = build_topology(topology_name)
        lengths = get_topology_link_lengths(topology_name)
        topology_signatures[topology_name] = {
            "nodes": sorted(nodes),
            "edges": [
                [u, v, float(lengths[edge_key(u, v)])]
                for u, v in sorted(edge_key(u, v) for u, v in edges)
            ],
        }

    return {
        "experiment_tag": EXPERIMENT_TAG,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "snapshot_policy": "post_arrival_only",
        "fallback_debug_policy": "one_row_per_steady_state_fallback_invocation",
        "traffic_trace_policy": "sha256_of_canonical_generated_request_sequence_v1",
        "decision_time_scope": "route_and_spectrum_decision_only_perf_counter_ns",
        "slot_hop_cost_definition": "sum_allocated_segment_width_times_path_hops",
        "carried_traffic_definition": "mean_sum_active_accepted_request_gbps_at_post_arrival_snapshots",
        "carried_efficiency_definition": "mean_active_carried_gbps_divided_by_mean_occupied_slot_hops",
        "physical_layer_scope": "rsa_only_no_qot_reach_guard_band_transponder_delay_or_control_plane_model",
        "aggregate_offered_load_definition": "nominal_load_index_times_topology_load_scale",
        "total_slots": TOTAL_SLOTS,
        "slot_granularity_ghz": SLOT_GRANULARITY_GHZ,
        "spectrum_bandwidth_ghz": SPECTRUM_BANDWIDTH_GHZ,
        "bpsk_m": BPSK_M,
        "load_values": list(LOAD_VALUES),
        "n_requests": N_REQUESTS,
        "warmup_requests": WARMUP_REQUESTS,
        "seeds": list(SEEDS),
        "bandwidth_min_gbps": BANDWIDTH_MIN_GBPS,
        "bandwidth_max_gbps": BANDWIDTH_MAX_GBPS,
        "bandwidth_sampling_mode": BANDWIDTH_SAMPLING_MODE,
        "service_rate": SERVICE_RATE,
        "load_scale": LOAD_SCALE,
        "load_scale_by_topology": dict(sorted(LOAD_SCALE_BY_TOPOLOGY.items())),
        "topologies": list(TOPOLOGIES),
        "topology_signatures": topology_signatures,
        "schemes": list(SCHEMES),
        "default_k_max": DEFAULT_K_MAX,
        "max_segments_per_path": MAX_SEGMENTS_PER_PATH,
        "max_segments_per_request": MAX_SEGMENTS_PER_REQUEST,
        "main_redundancy_r": MAIN_REDUNDANCY_R,
        "mp_ss_k_max": MP_SS_K_MAX,
        "mp_ss_max_parts": MP_SS_MAX_PARTS,
        "fallback_segment_order": "width_descending_then_slot_index_ascending",
        "mp_ss_fallback_allocator": "deterministic_sequential_greedy_no_backtracking",
        "adaptive_fragment_allocator": "shared_by_adaptive_mp_ss_rsa_and_aca_rsa",
        "adaptive_mp_ss_semantics": "adaptive_uncoded_subdemands_r0_admission_equivalent_to_aca",
        "rb_scsp_implementation": "repair_oriented_proxy",
        "rb_scsp_max_repairs": RB_SCSP_MAX_REPAIRS,
    }


def ensure_checkpoint_configuration() -> None:
    """Create or validate a crash-safe configuration manifest before resuming."""
    config = scientific_configuration()
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    manifest = {
        "fingerprint_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "configuration": config,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = CHECKPOINT_CONFIG_JSON
    if not manifest_path.exists():
        existing_manifests = sorted(RESULTS_DIR.glob("checkpoint_configuration_*.json"))
        existing_checkpoints = list(RESULTS_DIR.glob("checkpoint_*"))
        if existing_manifests:
            if len(existing_manifests) != 1:
                raise RuntimeError("Multiple checkpoint configuration manifests were found.")
            manifest_path = existing_manifests[0]
        elif existing_checkpoints:
            raise RuntimeError(
                "Checkpoint rows exist without a configuration manifest; refusing an unsafe resume."
            )

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing.get("fingerprint_sha256") != manifest["fingerprint_sha256"]:
            raise RuntimeError(
                "Checkpoint configuration mismatch. Use a new results directory or restore "
                "the exact simulator and scientific configuration that created this checkpoint."
            )
        print(f"Checkpoint configuration verified: {manifest_path}", flush=True)
        return

    temporary_path = CHECKPOINT_CONFIG_JSON.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, CHECKPOINT_CONFIG_JSON)
    print(f"Checkpoint configuration saved: {CHECKPOINT_CONFIG_JSON}", flush=True)

def _worker_shard_path(
    shard_dir: Path,
    kind: str,
    topology: str,
    scheme: str,
    load: float,
    seed: int,
) -> Path:
    """Return a deterministic, run-unique worker shard path."""
    load_token = f"{float(load):g}".replace(".", "p")
    return shard_dir / f"{kind}_{topology}_{scheme}_load_{load_token}_seed_{seed}.csv"


def _write_worker_shard(rows: List[Dict[str, object]], path: Path) -> Optional[str]:
    """Atomically publish one worker's rows without touching shared checkpoints."""
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    save_csv(rows, temporary_path)
    os.replace(temporary_path, path)
    return str(path)


def run_seed_worker(
    job: Tuple[str, str, float, int],
    shard_dir_text: str,
) -> Dict[str, object]:
    """Run one independent seed simulation inside a worker process."""
    topology_name, scheme, load, seed = job
    sim = ComparativeSimulator(
        scheme=scheme,
        load_erlangs=load,
        seed=seed,
        topology_name=topology_name,
    )
    result = sim.run(
        n_requests=N_REQUESTS,
        warmup_requests=WARMUP_REQUESTS,
    )

    shard_dir = Path(shard_dir_text)
    debug_shard = None
    input_shard = None
    if ENABLE_K_DEBUG_LOG and sim.k_debug_rows:
        debug_shard = _write_worker_shard(
            sim.k_debug_rows,
            _worker_shard_path(
                shard_dir, "k_debug", topology_name, scheme, load, seed
            ),
        )
    if ENABLE_INPUT_DEMAND_LOG and sim.input_demand_rows:
        input_shard = _write_worker_shard(
            sim.input_demand_rows,
            _worker_shard_path(
                shard_dir, "input_demand", topology_name, scheme, load, seed
            ),
        )

    sim.k_debug_rows.clear()
    sim.input_demand_rows.clear()
    sim.active_requests.clear()
    sim.event_queue.clear()
    return {
        "result": result,
        "debug_shard": debug_shard,
        "input_shard": input_shard,
    }


def _delete_worker_shard(path_text: Optional[str]) -> None:
    if path_text:
        Path(path_text).unlink(missing_ok=True)


def commit_worker_result(
    payload: Dict[str, object],
    rows: List[Dict[str, object]],
    completed_runs: set,
    raw_checkpoint_csv: Path,
    k_debug_checkpoint_csv: Path,
    input_checkpoint_csv: Path,
    completed_checkpoint_csv: Path,
) -> Dict[str, object]:
    """Commit a completed worker result through the single parent writer."""
    result = payload["result"]
    if not isinstance(result, dict):
        raise TypeError("Worker returned an invalid result payload.")

    run_key = make_run_key(
        result["topology"], result["scheme"], result["load"], result["seed"]
    )
    if run_key in completed_runs:
        _delete_worker_shard(payload.get("debug_shard"))
        _delete_worker_shard(payload.get("input_shard"))
        return result

    # The completion marker is deliberately written last. If interruption
    # occurs earlier, prepare_checkpoint_resume() removes the uncommitted rows.
    append_csv([result], raw_checkpoint_csv)
    append_csv_shard(payload.get("debug_shard"), k_debug_checkpoint_csv)
    append_csv_shard(payload.get("input_shard"), input_checkpoint_csv)
    rows.append(result)
    save_csv(summarize_results(rows), CHECKPOINT_SUMMARY_CSV)
    append_csv(
        [{
            "topology": result["topology"],
            "scheme": result["scheme"],
            "load": result["load"],
            "seed": result["seed"],
        }],
        completed_checkpoint_csv,
    )
    completed_runs.add(run_key)
    _delete_worker_shard(payload.get("debug_shard"))
    _delete_worker_shard(payload.get("input_shard"))
    return result


def run_all_experiments(max_workers: int) -> Tuple[
    List[Dict[str, float]], 
    List[Dict[str, object]], 
    List[Dict[str, object]]
]:
    ensure_checkpoint_configuration()
    raw_checkpoint_csv = resolve_checkpoint_path(
        CHECKPOINT_RAW_CSV,
        "checkpoint_raw_per_seed_*.csv",
    )
    k_debug_checkpoint_csv = resolve_checkpoint_path(
        CHECKPOINT_K_DEBUG_CSV,
        "checkpoint_k_debug_per_seed_*.csv",
    )
    input_checkpoint_csv = resolve_checkpoint_path(
        CHECKPOINT_INPUT_DEMAND_CSV,
        "checkpoint_input_demand_per_seed_*.csv",
    )
    completed_checkpoint_csv = resolve_checkpoint_path(
        CHECKPOINT_COMPLETED_RUNS_CSV,
        "checkpoint_completed_runs_*.csv",
    )
    if RESUME_FROM_CHECKPOINT:
        rows, completed_runs = prepare_checkpoint_resume(
            raw_checkpoint_csv,
            k_debug_checkpoint_csv,
            input_checkpoint_csv,
            completed_checkpoint_csv,
        )
    else:
        rows, completed_runs = [], set()

    print("\n=== Comparative EON RSA / ACA-RSA Study ===")
    print("Schemes:")
    print("  - k1_rsa        : static K=1 baseline")
    print("  - k3_rsa        : static K=3 baseline")
    print("  - fsa_rsa_ssfm  : dynamic SSFM single-path baseline")
    print("  - fsa_k3_rsa    : dynamic SSFM top-3 baseline")
    print("  - rb_scsp_arsa  : repair-enhanced heuristic baseline")
    print("  - mp_ss_rsa     : fixed-partition multipath split-spectrum without coding")
    print("  - adaptive_mp_ss_rsa: adaptive non-coding fragment aggregation")
    print("  - aca_rsa       : dynamic SSFM single-path + coding fallback")
    print("  - p_aca_rsa     : proactive coding-aware fragmented allocation")
    print("  - aca_k3_rsa    : dynamic SSFM top-3 + coding fallback")
    print()

    pending_jobs: List[Tuple[str, str, float, int]] = []
    for topology_name in TOPOLOGIES:
        for scheme in SCHEMES:
            for load in LOAD_VALUES:
                for seed in SEEDS:
                    run_key = make_run_key(topology_name, scheme, load, seed)
                    if run_key in completed_runs:
                        print(
                            f"topology={topology_name:<8} "
                            f"scheme={scheme:<14} "
                            f"load={load:>3} "
                            f"seed={seed} "
                            f"| skipped, already in checkpoint",
                            flush=True,
                        )
                    else:
                        pending_jobs.append((topology_name, scheme, load, seed))

    if not pending_jobs:
        print("All configured runs are already complete.", flush=True)
        return rows, [], []

    worker_count = min(max(1, max_workers), len(pending_jobs))
    shard_dir = RESULTS_DIR / ".parallel_worker_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Parallel execution: {len(pending_jobs)} pending seed runs "
        f"with {worker_count} worker processes.",
        flush=True,
    )

    failures: List[Tuple[Tuple[str, str, float, int], str]] = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(run_seed_worker, job, str(shard_dir.resolve())): job
            for job in pending_jobs
        }
        committed = 0
        for future in as_completed(futures):
            job = futures[future]
            try:
                payload = future.result()
                result = commit_worker_result(
                    payload,
                    rows,
                    completed_runs,
                    raw_checkpoint_csv,
                    k_debug_checkpoint_csv,
                    input_checkpoint_csv,
                    completed_checkpoint_csv,
                )
            except Exception as exc:
                failures.append((job, f"{type(exc).__name__}: {exc}"))
                print(
                    f"topology={job[0]:<8} scheme={job[1]:<14} "
                    f"load={job[2]:>3} seed={job[3]} | FAILED: {exc}",
                    flush=True,
                )
                continue

            committed += 1
            print(
                f"[{committed}/{len(pending_jobs)}] "
                f"topology={result['topology']:<8} "
                f"scheme={result['scheme']:<14} "
                f"load={result['load']:>3} "
                f"seed={result['seed']} "
                f"RBP={result['RBP']:.5f} "
                f"BBP={result['BBP']:.5f} "
                f"throughput={result['throughput']:.5f} "
                f"FSU={result['FSU']:.4f} "
                f"K={result['avg_K']:.3f} "
                f"r/g={result['avg_r_over_g']:.3f} "
                f"decision_us={result['mean_decision_time_us']:.1f} "
                f"| checkpoint transaction committed",
                flush=True,
            )

    try:
        shard_dir.rmdir()
    except OSError:
        # Retain non-empty shards from interrupted/failed jobs for diagnosis.
        pass

    if failures:
        details = "; ".join(
            f"{topology}/{scheme}/load={load}/seed={seed}: {message}"
            for (topology, scheme, load, seed), message in failures
        )
        raise RuntimeError(
            f"{len(failures)} parallel seed run(s) failed. "
            f"Successful runs remain checkpointed. {details}"
        )

    return rows, [], []

def t_critical_95(num_samples: int) -> float:
    """
    Two-sided 95% Student-t critical value for seed-level confidence intervals.
    Falls back to the normal approximation for large sample counts.
    """
    if num_samples <= 1:
        return 0.0

    df = num_samples - 1
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        25: 2.060,
        30: 2.042,
    }
    if df in table:
        return table[df]
    if df < 25:
        return table[20]
    if df < 30:
        return table[25]
    return 1.96


def summarize_results(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[str, str, int], List[Dict[str, float]]] = {}
    for row in rows:
        key = (row["scheme"], row["topology"], int(row["load"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, float]] = []
    metrics = [
        "RBP",
        "BBP",
        "accepted_load",
        "throughput",
        "throughput_sum",
        "throughput_rate",
        "FSU",
        "avg_LFM",
        "FDM",
        "avg_EBFM",
        "avg_K",
        "avg_r_over_g",
        "nominal_load_index",
        "load_scale",
        "actual_aggregate_offered_load_erlangs",
        "actual_aggregate_accepted_load_erlangs",
        "accepted_request_count",
        "total_allocated_slot_hops",
        "mean_allocated_slot_hops_per_accepted_request",
        "slot_hops_per_accepted_gbps",
        "accepted_gbps_per_slot_hop",
        "mean_occupied_slot_hops",
        "mean_active_carried_gbps",
        "carried_gbps_per_occupied_slot_hop",
        "occupied_slot_hops_per_carried_gbps",
        "decision_sample_count",
        "mean_decision_time_us",
        "median_decision_time_us",
        "p95_decision_time_us",
        "simulation_runtime_seconds",
    ]

    for (scheme, topology, load), group in sorted(grouped.items()):
        summary = {
            "scheme": scheme,
            "topology": topology,
            "load": load,
            "num_seeds": len(group),
        }
        for metric in metrics:
            vals = [float(item[metric]) for item in group]
            mu = mean(vals)
            sigma = stdev(vals) if len(vals) > 1 else 0.0
            ci95 = (
                t_critical_95(len(vals)) * sigma / math.sqrt(len(vals))
                if len(vals) > 1 else 0.0
            )
            summary[f"{metric}_mean"] = mu
            summary[f"{metric}_std"] = sigma
            summary[f"{metric}_ci95"] = ci95
            summary[f"{metric}_ci95_low"] = mu - ci95
            summary[f"{metric}_ci95_high"] = mu + ci95
        summary_rows.append(summary)
    return summary_rows


# ============================================================
# CSV output
# ============================================================

def make_run_key(topology: object, scheme: object, load: object, seed: object) -> Tuple[str, str, int, int]:
    return (str(topology), str(scheme), int(float(load)), int(seed))


def completed_run_keys(rows: List[Dict[str, object]]) -> set:
    keys = set()
    for row in rows:
        try:
            keys.add(make_run_key(row["topology"], row["scheme"], row["load"], row["seed"]))
        except (KeyError, TypeError, ValueError):
            continue
    return keys


def remove_incomplete_checkpoint_rows(path: Path, incomplete_runs: set, deduplicate: bool = False) -> int:
    """Atomically remove rows belonging to uncommitted runs without loading the file."""
    if not path.exists() or path.stat().st_size == 0:
        return 0

    temporary_path = path.with_name(path.name + ".cleanup.tmp")
    removed = 0
    seen = set()
    with path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            return 0
        with temporary_path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                try:
                    key = make_run_key(row["topology"], row["scheme"], row["load"], row["seed"])
                except (KeyError, TypeError, ValueError):
                    removed += 1
                    continue
                if key in incomplete_runs or (deduplicate and key in seen):
                    removed += 1
                    continue
                seen.add(key)
                writer.writerow(row)
            target.flush()
            os.fsync(target.fileno())
    os.replace(temporary_path, path)
    return removed


def prepare_checkpoint_resume(
    raw_path: Path,
    debug_path: Path,
    input_path: Path,
    completed_path: Path,
) -> Tuple[List[Dict[str, object]], set]:
    """Recover only fully committed runs and purge any interrupted transaction."""
    completion_rows = load_csv_rows(completed_path)
    completed_runs = completed_run_keys(completion_rows)
    raw_rows = load_csv_rows(raw_path)
    raw_run_keys = completed_run_keys(raw_rows)

    orphan_markers = completed_runs - raw_run_keys
    if orphan_markers:
        raise RuntimeError(
            "Completion checkpoint contains run keys missing from the raw checkpoint; "
            "refusing an unsafe resume."
        )

    incomplete_runs = raw_run_keys - completed_runs
    raw_needs_cleanup = bool(incomplete_runs) or len(raw_rows) != len(raw_run_keys)
    if raw_needs_cleanup:
        removed_raw = remove_incomplete_checkpoint_rows(raw_path, incomplete_runs, deduplicate=True)
        removed_debug = remove_incomplete_checkpoint_rows(debug_path, incomplete_runs)
        removed_input = remove_incomplete_checkpoint_rows(input_path, incomplete_runs)
        print(
            "Recovered interrupted checkpoint transaction(s): "
            f"removed raw={removed_raw}, debug={removed_debug}, input={removed_input} stale rows.",
            flush=True,
        )
        raw_rows = load_csv_rows(raw_path)

    return raw_rows, completed_runs


def load_csv_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_checkpoint_path(expected_path: Path, pattern: str) -> Path:
    """
    Use the expected checkpoint path when it exists. When resuming from an older
    results folder whose experiment tag differs, fall back to the newest matching
    checkpoint in that folder so completed seed runs are not lost.
    """
    if expected_path.exists():
        return expected_path

    candidates = sorted(
        expected_path.parent.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        print(f"Using existing checkpoint for resume/finalization: {candidates[0]}")
        return candidates[0]
    return expected_path


def save_csv(rows: List[Dict[str, float]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())

def append_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def append_csv_shard(source_path: object, destination_path: Path) -> None:
    """Stream one complete worker shard into a parent-owned checkpoint CSV."""
    if not source_path:
        return
    source = Path(str(source_path))
    if not source.exists() or source.stat().st_size == 0:
        return

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_has_rows = destination_path.exists() and destination_path.stat().st_size > 0

    with source.open("rb") as source_stream:
        source_header = source_stream.readline()
        if not source_header:
            return

        if destination_has_rows:
            with destination_path.open("rb") as destination_reader:
                destination_header = destination_reader.readline()
            if destination_header.rstrip(b"\r\n") != source_header.rstrip(b"\r\n"):
                raise RuntimeError(
                    f"Worker shard header does not match checkpoint header: {source}"
                )

        with destination_path.open("ab") as destination_stream:
            if not destination_has_rows:
                destination_stream.write(source_header)
            shutil.copyfileobj(source_stream, destination_stream, length=1024 * 1024)
            destination_stream.flush()
            os.fsync(destination_stream.fileno())


def result_row_sort_key(row: Dict[str, object]) -> Tuple[int, int, float, int]:
    """Provide deterministic final CSV ordering independent of worker completion."""
    topology_rank = {name: index for index, name in enumerate(TOPOLOGIES)}
    scheme_rank = {name: index for index, name in enumerate(SCHEMES)}
    return (
        topology_rank.get(str(row.get("topology")), len(topology_rank)),
        scheme_rank.get(str(row.get("scheme")), len(scheme_rank)),
        float(row.get("load", 0)),
        int(float(row.get("seed", 0))),
    )


def validate_paired_traffic_traces(rows: List[Dict[str, object]]) -> int:
    """Require one offered-traffic fingerprint per topology/load/seed group."""
    grouped: Dict[Tuple[str, int, int], set] = {}
    for row in rows:
        key = (
            str(row["topology"]),
            int(float(row["load"])),
            int(float(row["seed"])),
        )
        trace_hash = str(row.get("traffic_trace_sha256", "")).strip()
        if not trace_hash:
            raise RuntimeError(f"Missing traffic trace hash for paired group {key}.")
        grouped.setdefault(key, set()).add(trace_hash)

    mismatches = {key: hashes for key, hashes in grouped.items() if len(hashes) != 1}
    if mismatches:
        first_key = next(iter(mismatches))
        raise RuntimeError(
            "Paired-traffic validation failed: schemes received different request "
            f"traces for {first_key}."
        )
    return len(grouped)


def validate_adaptive_r0_equivalence(rows: List[Dict[str, object]]) -> int:
    """Verify the designed r=0 admission/resource equivalence of ACA and A-MP-SS."""
    paired: Dict[Tuple[str, int, int], Dict[str, Dict[str, object]]] = {}
    for row in rows:
        scheme = str(row.get("scheme"))
        if scheme not in {"aca_rsa", "adaptive_mp_ss_rsa"}:
            continue
        key = (
            str(row["topology"]),
            int(float(row["load"])),
            int(float(row["seed"])),
        )
        paired.setdefault(key, {})[scheme] = row

    metrics = [
        "RBP",
        "BBP",
        "throughput",
        "throughput_sum",
        "throughput_rate",
        "FSU",
        "FDM",
        "avg_EBFM",
        "avg_K",
        "accepted_request_count",
        "total_allocated_slot_hops",
        "mean_allocated_slot_hops_per_accepted_request",
        "slot_hops_per_accepted_gbps",
        "accepted_gbps_per_slot_hop",
        "mean_occupied_slot_hops",
        "mean_active_carried_gbps",
        "carried_gbps_per_occupied_slot_hop",
        "occupied_slot_hops_per_carried_gbps",
    ]
    validated = 0
    for key, scheme_rows in paired.items():
        if set(scheme_rows) != {"aca_rsa", "adaptive_mp_ss_rsa"}:
            raise RuntimeError(f"Incomplete ACA/A-MP-SS comparison for {key}.")
        aca = scheme_rows["aca_rsa"]
        adaptive = scheme_rows["adaptive_mp_ss_rsa"]
        for metric in metrics:
            aca_value = float(aca[metric])
            adaptive_value = float(adaptive[metric])
            if not math.isclose(aca_value, adaptive_value, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(
                    "ACA/A-MP-SS r=0 equivalence failed for "
                    f"{key}, metric={metric}: {aca_value} != {adaptive_value}."
                )
        validated += 1
    return validated

def filter_rows_by_topology(rows, topology_name: str):
    return [row for row in rows if row.get("topology") == topology_name]

# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run independent EON seed simulations in parallel."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=(
            "Number of worker processes. Default: "
            f"{DEFAULT_MAX_WORKERS} (bounded by available pending runs)."
        ),
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def main(max_workers: int = DEFAULT_MAX_WORKERS) -> None:
    initialize_output_paths()
    raw_rows, _, _ = run_all_experiments(max_workers=max_workers)

    # Reload from the checkpoint so final outputs include every completed seed,
    # including rows saved before any interrupted/resumed run.
    raw_checkpoint_csv = resolve_checkpoint_path(
        CHECKPOINT_RAW_CSV,
        "checkpoint_raw_per_seed_*.csv",
    )
    checkpoint_rows = load_csv_rows(raw_checkpoint_csv)
    if checkpoint_rows:
        raw_rows = checkpoint_rows
    raw_rows = sorted(raw_rows, key=result_row_sort_key)

    paired_group_count = validate_paired_traffic_traces(raw_rows)
    adaptive_pair_count = validate_adaptive_r0_equivalence(raw_rows)

    # Memory-safe finalization: do not reload large debug/input checkpoints.
    # They are retained as per-seed checkpoint CSVs for separate diagnostics.
    debug_rows = []
    input_demand_rows = []

    # Save combined raw results
    save_csv(raw_rows, RAW_RESULTS_CSV)

    # Save combined summary results
    summary_rows = summarize_results(raw_rows)
    save_csv(summary_rows, SUMMARY_RESULTS_CSV)

    # Save per-topology raw and summary results
    for topology_name in TOPOLOGIES:
        topo_raw_rows = filter_rows_by_topology(raw_rows, topology_name)
        topo_summary_rows = filter_rows_by_topology(summary_rows, topology_name)

        save_csv(
            topo_raw_rows,
            RESULTS_DIR / f"simulation_results_raw_{EXPERIMENT_TAG}_{topology_name}.csv"
        )

        save_csv(
            topo_summary_rows,
            RESULTS_DIR / f"simulation_results_summary_{EXPERIMENT_TAG}_{topology_name}.csv"
        )

        if ENABLE_K_DEBUG_LOG and SAVE_PER_TOPOLOGY_K_DEBUG_CSV and debug_rows:
            topo_debug_rows = filter_rows_by_topology(debug_rows, topology_name)
            save_csv(
                topo_debug_rows,
                RESULTS_DIR / f"k_debug_{EXPERIMENT_TAG}_{topology_name}.csv"
            )

        if input_demand_rows:
            topo_input_rows = filter_rows_by_topology(input_demand_rows, topology_name)
            save_csv(
                topo_input_rows,
                RESULTS_DIR / f"input_demand_{EXPERIMENT_TAG}_{topology_name}.csv"
            )

    # if input_demand_rows:
    #     save_csv(input_demand_rows, INPUT_DEMAND_CSV)
    #     print(f"Saved combined input demand log to: {INPUT_DEMAND_CSV}")

    if ENABLE_K_DEBUG_LOG and SAVE_FINAL_K_DEBUG_CSV and debug_rows:
        save_csv(debug_rows, K_DEBUG_CSV)
        print(f"Saved combined K debug log to: {K_DEBUG_CSV}")

    print(f"\nSaved combined raw results to: {RAW_RESULTS_CSV}")
    print(f"Saved combined summary results to: {SUMMARY_RESULTS_CSV}")
    print(f"Per-seed checkpoint raw results were saved during the run to: {raw_checkpoint_csv}")
    print(f"Rolling checkpoint summary was saved during the run to: {CHECKPOINT_SUMMARY_CSV}")
    print(f"Verified identical offered traffic for {paired_group_count} paired run groups.")
    print(f"Verified ACA/A-MP-SS r=0 equivalence for {adaptive_pair_count} run pairs.")
    if ENABLE_K_DEBUG_LOG and not SAVE_FINAL_K_DEBUG_CSV:
        k_debug_checkpoint_csv = resolve_checkpoint_path(
            CHECKPOINT_K_DEBUG_CSV,
            "checkpoint_k_debug_per_seed_*.csv",
        )
        print(f"K debug checkpoint retained without loading into memory: {k_debug_checkpoint_csv}")
    print(f"Saved per-topology CSV files in: {RESULTS_DIR}")

if __name__ == "__main__":
    mp.freeze_support()
    command_line_args = parse_args()
    main(max_workers=command_line_args.workers)
