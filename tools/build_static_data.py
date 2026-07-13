#!/usr/bin/env python3
"""Build the browser-only masterclass dataset from the reduced ROOT file.

The deployed site never executes Python.  This script is an offline build step
that converts the required ROOT branches into compact, gzip-compressed binary
typed arrays plus a small JSON manifest consumed by ``static/data-engine.js``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import uproot


APP_VERSION = "4.0.0"
COLLECTION_ORDER = ("Events", "Leptons", "PairingOptions", "QuadLeptons")

# Only columns used by the current teaching interface are shipped to each
# browser.  Derived columns are added below before this selection is applied.
COLUMNS: dict[str, tuple[str, ...] | None] = {
    "Events": None,  # Small after compression, and useful for event provenance.
    "Leptons": (
        "event_index", "lepton_index", "abs_pdgid", "charge",
        "pt", "eta", "phi", "energy",
    ),
    "PairingOptions": (
        "event_index", "quad_index", "quad_is_best", "pairing_id",
        "pair_a_lep1_index", "pair_a_lep2_index",
        "pair_b_lep1_index", "pair_b_lep2_index",
        "pair_a_mass", "pair_a_mass_squared", "pair_a_pt", "pair_a_energy",
        "pair_a_px", "pair_a_py", "pair_a_pz", "pair_a_momentum",
        "pair_a_pt_scalar", "pair_a_charge_sum", "pair_a_dr", "pair_a_dphi",
        "pair_a_opening_angle", "pair_a_is_os", "pair_a_is_sf", "pair_a_is_ss",
        "pair_b_mass", "pair_b_mass_squared", "pair_b_pt", "pair_b_energy",
        "pair_b_px", "pair_b_py", "pair_b_pz", "pair_b_momentum",
        "pair_b_pt_scalar", "pair_b_charge_sum", "pair_b_dr", "pair_b_dphi",
        "pair_b_opening_angle", "pair_b_is_os", "pair_b_is_sf", "pair_b_is_ss",
    ),
    "QuadLeptons": (
        "event_index", "quad_index", "lep1_index", "lep2_index",
        "lep3_index", "lep4_index", "channel", "charge_sum", "n_ossf_pairs",
        "is_zz_candidate", "is_best", "mass", "mass_squared", "pt", "energy",
        "px", "py", "pz", "momentum", "pt_scalar", "dr", "opening_angle",
        "leading_lepton_pt", "subleading_lepton_pt", "min_lepton_pt",
        "max_abs_lepton_eta", "max_lepton_iso", "z1_mass", "z2_mass",
        "abs_z1_mass_minus_z", "abs_z2_mass_minus_z", "has_ossf_pairing",
        "pass_pt_hzz", "pass_isolation_035", "pass_z_windows",
    ),
}

JS_TYPES = {
    "<f4": "Float32Array",
    "<f8": "Float64Array",
    "<i4": "Int32Array",
    "<u4": "Uint32Array",
    "|u1": "Uint8Array",
}


def load_collections(root_path: Path) -> dict[str, dict[str, np.ndarray]]:
    data: dict[str, dict[str, np.ndarray]] = {}
    with uproot.open(root_path) as root_file:
        for collection in COLLECTION_ORDER:
            if collection not in root_file:
                raise KeyError(f"Required ROOT collection {collection!r} is missing")
            arrays = root_file[collection].arrays(library="np")
            data[collection] = {name: np.asarray(array) for name, array in arrays.items()}
    enrich_quad_fields(data)
    enrich_pairing_fields(data)
    add_aliases(data)
    return data


def lepton_lookup(
    event_index: np.ndarray,
    lepton_index: np.ndarray,
    query_events: np.ndarray,
    query_indices: np.ndarray,
) -> np.ndarray:
    factor = np.int64(10_000)
    keys = event_index.astype(np.int64) * factor + lepton_index.astype(np.int64)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    query = query_events.astype(np.int64) * factor + query_indices.astype(np.int64)
    positions = np.searchsorted(sorted_keys, query)
    if (
        len(sorted_keys) == 0
        or np.any(positions >= len(sorted_keys))
        or np.any(sorted_keys[np.minimum(positions, len(sorted_keys) - 1)] != query)
    ):
        raise ValueError("Lepton lookup mismatch while deriving browser data")
    return order[positions]


def enrich_quad_fields(data: dict[str, dict[str, np.ndarray]]) -> None:
    quads = data["QuadLeptons"]
    leptons = data["Leptons"]
    matched = [
        lepton_lookup(
            leptons["event_index"], leptons["lepton_index"],
            quads["event_index"], quads[name],
        )
        for name in ("lep1_index", "lep2_index", "lep3_index", "lep4_index")
    ]

    pts = np.stack([leptons["pt"][index] for index in matched], axis=1)
    ordered_pts = np.sort(pts, axis=1)[:, ::-1]
    etas = np.stack([leptons["eta"][index] for index in matched], axis=1)
    phis = np.stack([leptons["phi"][index] for index in matched], axis=1)
    isolations = np.stack([leptons["iso"][index] for index in matched], axis=1)

    quads.setdefault("leading_lepton_pt", ordered_pts[:, 0].astype(np.float32))
    quads.setdefault("subleading_lepton_pt", ordered_pts[:, 1].astype(np.float32))
    quads.setdefault("min_lepton_pt", ordered_pts[:, -1].astype(np.float32))
    quads.setdefault("max_abs_lepton_eta", np.max(np.abs(etas), axis=1).astype(np.float32))
    quads.setdefault("max_lepton_iso", np.max(isolations, axis=1).astype(np.float32))

    pair_indices = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    separations: list[np.ndarray] = []
    angles: list[np.ndarray] = []
    transverse = 1.0 / np.cosh(etas)
    directions = np.stack(
        [transverse * np.cos(phis), transverse * np.sin(phis), np.tanh(etas)],
        axis=2,
    )
    for first, second in pair_indices:
        deta = etas[:, first] - etas[:, second]
        dphi = (phis[:, first] - phis[:, second] + np.pi) % (2 * np.pi) - np.pi
        separations.append(np.sqrt(deta * deta + dphi * dphi))
        dot = np.sum(directions[:, first, :] * directions[:, second, :], axis=1)
        angles.append(np.arccos(np.clip(dot, -1.0, 1.0)))
    quads.setdefault("dr", np.mean(np.stack(separations, axis=1), axis=1).astype(np.float32))
    quads.setdefault("opening_angle", np.mean(np.stack(angles, axis=1), axis=1).astype(np.float32))
    quads.setdefault("pass_pt_hzz", ((ordered_pts[:, 0] > 20) & (ordered_pts[:, 1] > 10)).astype(np.int32))
    max_iso = np.max(isolations, axis=1)
    quads.setdefault("pass_isolation_035", ((max_iso < 0.35) | (max_iso < 0)).astype(np.int32))
    quads.setdefault(
        "pass_z_windows",
        ((quads["z1_mass"] > 40) & (quads["z1_mass"] < 120)
         & (quads["z2_mass"] > 12) & (quads["z2_mass"] < 120)).astype(np.int32),
    )
    quads.setdefault("has_ossf_pairing", (quads["n_ossf_pairs"] >= 2).astype(np.int32))


def enrich_pairing_fields(data: dict[str, dict[str, np.ndarray]]) -> None:
    pairings = data["PairingOptions"]
    leptons = data["Leptons"]
    for prefix in ("pair_a", "pair_b"):
        first = lepton_lookup(
            leptons["event_index"], leptons["lepton_index"],
            pairings["event_index"], pairings[f"{prefix}_lep1_index"],
        )
        second = lepton_lookup(
            leptons["event_index"], leptons["lepton_index"],
            pairings["event_index"], pairings[f"{prefix}_lep2_index"],
        )
        q1, q2 = leptons["charge"][first], leptons["charge"][second]
        p1, p2 = leptons["abs_pdgid"][first], leptons["abs_pdgid"][second]
        pairings.setdefault(f"{prefix}_charge_sum", (q1 + q2).astype(np.int32))
        pairings.setdefault(f"{prefix}_is_sf", (p1 == p2).astype(np.int32))
        pairings.setdefault(f"{prefix}_is_os", (q1 != q2).astype(np.int32))
        pairings.setdefault(f"{prefix}_is_ss", (q1 == q2).astype(np.int32))


def add_aliases(data: dict[str, dict[str, np.ndarray]]) -> None:
    quads = data["QuadLeptons"]
    if "scalar_pt_sum" in quads:
        quads.setdefault("pt_scalar", quads["scalar_pt_sum"])
    pairings = data["PairingOptions"]
    for prefix in ("pair_a", "pair_b"):
        source = f"{prefix}_scalar_pt_sum"
        if source in pairings:
            pairings.setdefault(f"{prefix}_pt_scalar", pairings[source])


def browser_array(array: np.ndarray) -> tuple[np.ndarray, str]:
    """Return a portable little-endian array and its JavaScript constructor."""
    kind = array.dtype.kind
    if kind == "f":
        dtype = np.dtype("<f4" if array.dtype.itemsize <= 4 else "<f8")
    elif kind == "i" and array.dtype.itemsize <= 4:
        dtype = np.dtype("<i4")
    elif kind == "u" and array.dtype.itemsize <= 4:
        dtype = np.dtype("<u4")
    elif kind in {"i", "u"}:
        # JavaScript BigInt arrays complicate arithmetic and JSON conversion.
        # These identifiers are exactly representable by Number for this file.
        maximum = int(np.max(np.abs(array.astype(object)))) if len(array) else 0
        if maximum >= 2**53:
            raise ValueError(f"Integer value {maximum} cannot be represented exactly in JavaScript")
        dtype = np.dtype("<f8")
    elif kind == "b":
        dtype = np.dtype("u1")
    else:
        raise TypeError(f"Unsupported branch dtype {array.dtype}")
    converted = np.ascontiguousarray(array, dtype=dtype)
    key = converted.dtype.str
    return converted, JS_TYPES[key]


def write_collection(
    name: str,
    arrays: dict[str, np.ndarray],
    destination: Path,
) -> dict[str, Any]:
    requested = COLUMNS[name]
    names = list(arrays) if requested is None else list(requested)
    missing = [column for column in names if column not in arrays]
    if missing:
        raise KeyError(f"{name} is missing required columns: {', '.join(missing)}")

    # Event rows must be contiguous for binary-search slicing in the worker.
    order = np.argsort(arrays["event_index"], kind="stable")
    pieces: list[bytes] = []
    columns: dict[str, dict[str, Any]] = {}
    offset = 0
    for column in names:
        converted, js_type = browser_array(arrays[column][order])
        alignment = converted.dtype.itemsize
        padding = (-offset) % alignment
        if padding:
            pieces.append(bytes(padding))
            offset += padding
        raw = converted.tobytes(order="C")
        columns[column] = {
            "type": js_type,
            "offset": offset,
            "length": int(len(converted)),
        }
        pieces.append(raw)
        offset += len(raw)

    uncompressed = b"".join(pieces)
    compressed = gzip.compress(uncompressed, compresslevel=9, mtime=0)
    filename = f"{name}.dat"
    (destination / filename).write_bytes(compressed)
    return {
        "file": filename,
        "compression": "gzip",
        "rows": int(len(order)),
        "uncompressed_bytes": len(uncompressed),
        "compressed_bytes": len(compressed),
        "sha256": hashlib.sha256(compressed).hexdigest(),
        "columns": columns,
    }


def public_metadata(
    source_metadata: dict[str, Any],
    requirements: list[dict[str, Any]],
    collection_specs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pair_fields = set(collection_specs["PairingOptions"]["columns"])
    quad_fields = set(collection_specs["QuadLeptons"]["columns"])
    catalog = []
    for definition in requirements:
        fields = set(definition.get("fields", []))
        available = any(
            (mode == "pairs" and fields <= pair_fields)
            or (mode == "four" and fields <= quad_fields)
            for mode in definition["modes"]
        )
        catalog.append({key: value for key, value in definition.items() if key != "fields"} | {"available": available})

    collections: dict[str, Any] = {}
    for name, spec in collection_specs.items():
        original = source_metadata.get("collections", {}).get(name, {})
        collections[name] = {
            **original,
            "name": name,
            "n_rows": spec["rows"],
            "variables": list(spec["columns"]),
        }

    return {
        "app_version": APP_VERSION,
        "runtime": "browser-only",
        "load_warnings": [],
        "metadata": source_metadata,
        "collections": collections,
        "requirement_catalog": catalog,
        "formula_variables": {
            "E": "total energy of the selected particles",
            "px": "combined x momentum",
            "py": "combined y momentum",
            "pz": "combined z momentum",
            "p": "combined 3D momentum magnitude",
            "pt": "combined transverse momentum",
            "charge": "total charge",
            "pt_scalar": "scalar sum of the individual lepton pT values",
            "dr": "angular separation for a pair, or mean pairwise separation for four leptons",
            "angle": "3D opening angle for a pair, or mean pairwise angle for four leptons",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_file", type=Path, help="Reduced masterclass ROOT file")
    parser.add_argument("metadata_file", type=Path, help="Masterclass metadata JSON")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[1] / "site" / "data",
        help="Static data output directory",
    )
    parser.add_argument(
        "--requirements", type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "requirements.json",
        help="Event-requirement catalogue",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_collections(args.root_file)
    specs = {
        name: write_collection(name, data[name], args.output)
        for name in COLLECTION_ORDER
    }
    source_metadata = json.loads(args.metadata_file.read_text(encoding="utf-8"))
    requirements = json.loads(args.requirements.read_text(encoding="utf-8"))
    manifest = {
        "format": "imperial-masterclass-typed-arrays-v1",
        "app_version": APP_VERSION,
        "event_display_url_template": "",
        "requirements": requirements,
        "collections": specs,
        "public_metadata": public_metadata(source_metadata, requirements, specs),
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    raw_total = sum(spec["uncompressed_bytes"] for spec in specs.values())
    compressed_total = sum(spec["compressed_bytes"] for spec in specs.values())
    print(f"Wrote {manifest_path}")
    for name, spec in specs.items():
        print(f"  {name:16s} {spec['rows']:9,d} rows  {spec['compressed_bytes'] / 2**20:7.2f} MiB")
    ratio = compressed_total / raw_total if raw_total else math.nan
    print(f"Total: {compressed_total / 2**20:.2f} MiB ({ratio:.1%} of raw typed arrays)")


if __name__ == "__main__":
    main()
