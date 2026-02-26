"""
Add fixed special-cluster descriptions at the beginning of a cluster_descriptions.json file.

Use as:
  python -m compressor_2.add_special_clusters /path/to/cluster_descriptions.json
"""

from __future__ import annotations

import argparse
import json


SPECIAL_CLUSTERS: list[dict] = [
    {
        "description": "EOS EOT",
        "example": ["<EOS>", "<EOT>"],
    },
    {
        "description": "Numbers, arithmetic operations (+, -, *, /), $, %, etc.",
        "example": ["0", "1", "2", "+", "-", "*", "/", "$", "%", "="],
    },
]


def add_special_clusters(path: str) -> None:
    """
    Read a cluster_descriptions JSON file, prepend the fixed special-cluster
    entries as "0" and "1", shift existing numeric keys by +2, and overwrite in-place.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    result: dict[str, dict] = {}
    result["0"] = SPECIAL_CLUSTERS[0]
    result["1"] = SPECIAL_CLUSTERS[1]
    for key, value in data.items():
        if key.isdigit():
            result[str(int(key) + 2)] = value
        else:
            result[key] = value

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepend special-cluster descriptions to a cluster_descriptions.json file (in-place)."
    )
    parser.add_argument(
        "file",
        help="Path to cluster_descriptions.json (modified in-place)",
    )
    args = parser.parse_args()
    add_special_clusters(args.file)


if __name__ == "__main__":
    main()
