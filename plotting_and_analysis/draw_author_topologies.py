"""
Draw the author-matrix topologies encoded in the packaged v31 simulator.

Outputs PNG files with node labels and link-length labels.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
SIM_PATH = PACKAGE_ROOT / "simulators" / "v31_main.py"
OUTDIR = PACKAGE_ROOT / "topologies" / "figures"

NODE_LABEL_FONT_SIZE = 11
EDGE_LABEL_FONT_SIZE = 11
NODE_SIZE_BY_TOPOLOGY = {
    "USNET": 900,
    "GERMAN": 900,
    "TI": 760,
}


def load_sim_module():
    spec = importlib.util.spec_from_file_location("aca_rsa_v31_topologies", SIM_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load simulator module from {SIM_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def draw_topology(sim, topology_name: str) -> None:
    nodes, edges = sim.build_topology(topology_name)
    lengths = sim.get_topology_link_lengths(topology_name)

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for u, v in edges:
        graph.add_edge(u, v, weight=lengths.get(sim.edge_key(u, v), 1.0))

    # Deterministic layout. Using distance as spring-layout edge weight gives
    # compact, readable drawings without requiring geographic coordinates.
    seed_by_topology = {"USNET": 24, "GERMAN": 17, "TI": 30}
    pos = nx.spring_layout(
        graph,
        seed=seed_by_topology.get(topology_name, 7),
        weight="weight",
        iterations=300,
        k=1.25 / (len(nodes) ** 0.5),
    )
    if topology_name == "TI" and 6 in pos:
        # Move node 6 to the right to reduce label overlap near nodes 21 and 24.
        pos[6] = pos[6] + [0.18, 0.0]

    fig_width = 13 if topology_name != "GERMAN" else 10
    fig_height = 9 if topology_name != "GERMAN" else 7
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        width=1.6,
        edge_color="#8aa9c7",
        alpha=0.95,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color="#e31a1c",
        edgecolors="white",
        linewidths=1.4,
        node_size=NODE_SIZE_BY_TOPOLOGY.get(topology_name, 900),
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        ax=ax,
        font_size=NODE_LABEL_FONT_SIZE,
        font_color="white",
        font_weight="bold",
    )

    edge_labels = {
        (u, v): str(int(data["weight"]))
        for u, v, data in graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        ax=ax,
        font_size=EDGE_LABEL_FONT_SIZE,
        font_color="#303030",
        bbox={
            "boxstyle": "round,pad=0.12",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.72,
        },
        rotate=False,
    )

    # Plot title intentionally omitted for manuscript-ready topology figures.
    # ax.set_title(
    #     f"{topology_name} Author Topology ({len(nodes)} nodes, {len(edges)} links)",
    #     fontsize=16,
    #     pad=16,
    # )
    ax.set_axis_off()
    fig.tight_layout()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / f"{topology_name.lower()}_author_topology.png", dpi=240)
    plt.close(fig)


def main() -> None:
    sim = load_sim_module()
    for topology_name in ["USNET", "GERMAN", "TI"]:
        draw_topology(sim, topology_name)
    print(f"Wrote topology plots to: {OUTDIR}")


if __name__ == "__main__":
    main()

