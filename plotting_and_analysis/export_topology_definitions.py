"""Export the v31 topology definitions to a portable JSON file."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
SIM_PATH = PACKAGE_ROOT / "simulators" / "v31_main.py"
OUTPUT_PATH = PACKAGE_ROOT / "topologies" / "topology_definitions.json"


def load_simulator():
    spec = importlib.util.spec_from_file_location("aca_rsa_v31_topology_export", SIM_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load simulator module from {SIM_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    sim = load_simulator()
    exported = {}
    for topology_name in ("GERMAN", "USNET", "TI"):
        nodes, edges = sim.build_topology(topology_name)
        lengths = sim.get_topology_link_lengths(topology_name)
        exported[topology_name] = {
            "nodes": list(nodes),
            "edges": [
                {
                    "u": u,
                    "v": v,
                    "length_km": lengths[sim.edge_key(u, v)],
                }
                for u, v in edges
            ],
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(exported, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote topology definitions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
