#!/usr/bin/env python3
"""
make_zz4l_masterclass.py

Convert a CMS NanoAOD/NANOAODSIM ZZ->4l ROOT file into a small,
student-facing ROOT file for a visual masterclass exercise.

Output collections:
  Events         one row per event
  Leptons        one row per selected electron or muon
  Dileptons      all selected 2-lepton combinations
  QuadLeptons    all selected 4-lepton combinations
  PairingOptions three possible pairings for every 4-lepton combination

The goal is not to make a publication analysis ntuple. The goal is to make
a friendly physics sandbox where students can explore:
  - Which lepton pairs look like Z bosons?
  - Why do opposite-sign same-flavor pairs matter?
  - How does a mass compare with energy or pT sums?
  - How do four leptons combine into a ZZ candidate?
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot
import vector

vector.register_awkward()

Z_MASS_GEV = 91.1876


def jagged_const_like(x: ak.Array, value: float | int) -> ak.Array:
    return ak.ones_like(x) * value


def jagged_true_like(x: ak.Array) -> ak.Array:
    return x == x


def to_numpy(x: Any, dtype: Any | None = None) -> np.ndarray:
    arr = ak.to_numpy(x)
    return arr.astype(dtype) if dtype is not None else arr


def flat_numpy(x: Any, dtype: Any | None = None) -> np.ndarray:
    arr = ak.to_numpy(ak.flatten(x, axis=None))
    return arr.astype(dtype) if dtype is not None else arr


def flat_event_index(jagged_quantity: ak.Array, event_index: np.ndarray) -> np.ndarray:
    return flat_numpy(ak.broadcast_arrays(jagged_quantity, event_index)[1], np.int64)


def one_if(mask: ak.Array) -> ak.Array:
    return ak.where(mask, 1, 0)


def momentum_mag(p4: ak.Array) -> ak.Array:
    return np.sqrt(p4.px * p4.px + p4.py * p4.py + p4.pz * p4.pz)


def mass_squared(p4: ak.Array) -> ak.Array:
    p = momentum_mag(p4)
    return p4.energy * p4.energy - p * p


def opening_angle(a: ak.Array, b: ak.Array) -> ak.Array:
    dot = a.px * b.px + a.py * b.py + a.pz * b.pz
    amag = momentum_mag(a)
    bmag = momentum_mag(b)
    cosang = dot / ak.where((amag * bmag) > 0, amag * bmag, 1.0)
    cosang = ak.where(cosang > 1.0, 1.0, ak.where(cosang < -1.0, -1.0, cosang))
    return np.arccos(cosang)


def write_if_nonempty(tree: uproot.writing.writable.WritableTree, payload: dict[str, np.ndarray]) -> None:
    if not payload:
        return
    n = len(next(iter(payload.values())))
    if n > 0:
        tree.extend(payload)


def choose_pairing_z_order(
    m_a: ak.Array,
    a_l1: ak.Array,
    a_l2: ak.Array,
    m_b: ak.Array,
    b_l1: ak.Array,
    b_l2: ak.Array,
) -> tuple[ak.Array, ak.Array, ak.Array, ak.Array, ak.Array, ak.Array]:
    """Within one pairing option, call the pair closer to the Z mass Z1."""
    a_is_z1 = abs(m_a - Z_MASS_GEV) <= abs(m_b - Z_MASS_GEV)
    z1_mass = ak.where(a_is_z1, m_a, m_b)
    z2_mass = ak.where(a_is_z1, m_b, m_a)
    z1_l1 = ak.where(a_is_z1, a_l1, b_l1)
    z1_l2 = ak.where(a_is_z1, a_l2, b_l2)
    z2_l1 = ak.where(a_is_z1, b_l1, a_l1)
    z2_l2 = ak.where(a_is_z1, b_l2, a_l2)
    return z1_mass, z2_mass, z1_l1, z1_l2, z2_l1, z2_l2


def make_metadata(output_root: Path, sample_name: str, sample_id: int, input_file: str) -> dict[str, Any]:
    return {
        "root_file": str(output_root),
        "input_file": input_file,
        "sample": {"sample_id": sample_id, "sample_name": sample_name},
        "z_mass_GeV": Z_MASS_GEV,
        "description": (
            "Simplified ZZ to four-lepton file for a visual masterclass. "
            "The file is intentionally small and pedagogical: it stores selected leptons, "
            "their combinations, and all possible ZZ pairing choices."
        ),
        "collections": {
            "Events": {
                "title": "Events",
                "description": "One row per event. Useful for best-candidate plots and event-level counts.",
                "default_variable": "best_m4l",
            },
            "Leptons": {
                "title": "Single leptons",
                "description": "One row per selected electron or muon.",
                "default_variable": "pt",
            },
            "Dileptons": {
                "title": "Two-lepton combinations",
                "description": "Every possible pair of selected leptons in an event.",
                "default_variable": "mass",
            },
            "QuadLeptons": {
                "title": "Four-lepton combinations",
                "description": "Every possible group of four selected leptons in an event.",
                "default_variable": "mass",
            },
            "PairingOptions": {
                "title": "ZZ pairing options",
                "description": "The three ways to split a four-lepton group into two pairs.",
                "default_variable": "score_looks_like_ZZ",
            },
        },
        "variables": {
            "pt": {"label": "transverse momentum", "unit": "GeV", "range": [0, 160], "tooltip": "Sideways momentum, measured perpendicular to the beam."},
            "eta": {"label": "eta", "unit": "", "range": [-3, 3], "tooltip": "Detector angle coordinate."},
            "phi": {"label": "phi", "unit": "rad", "range": [-math.pi, math.pi], "tooltip": "Azimuthal angle around the beam pipe."},
            "energy": {"label": "energy", "unit": "GeV", "range": [0, 500], "tooltip": "Total energy of the object or combination."},
            "momentum": {"label": "momentum magnitude", "unit": "GeV", "range": [0, 500], "tooltip": "Size of the 3D momentum vector."},
            "scalar_pt_sum": {"label": "sum of lepton pT", "unit": "GeV", "range": [0, 400], "tooltip": "Add the individual lepton pT values. A useful comparison, but not mass."},
            "mass": {"label": "combined mass", "unit": "GeV", "range": [0, 250], "tooltip": "The mass obtained after accounting for both energy and momentum."},
            "mass_squared": {"label": "mass squared", "unit": "GeV²", "range": [0, 60000], "tooltip": "Energy squared minus momentum squared."},
            "energy_minus_momentum": {"label": "energy minus momentum", "unit": "GeV", "range": [0, 120], "tooltip": "A simple attempt at removing motion from energy. It is not quite the invariant mass."},
            "dr": {"label": "angular separation ΔR", "unit": "", "range": [0, 6], "tooltip": "How far apart two leptons are in detector angle space."},
            "opening_angle": {"label": "3D opening angle", "unit": "rad", "range": [0, math.pi], "tooltip": "The angle between two leptons in three dimensions."},
            "charge_sum": {"label": "charge sum", "unit": "", "range": [-4, 4], "tooltip": "Total electric charge of the combination."},
            "is_os": {"label": "opposite charge", "unit": "", "tooltip": "1 means the two leptons have opposite electric charge."},
            "is_sf": {"label": "same flavor", "unit": "", "tooltip": "1 means the two leptons are both electrons or both muons."},
            "is_ossf": {"label": "opposite-charge same-flavor", "unit": "", "tooltip": "Often useful when studying Z-boson decays."},
            "best_m4l": {"label": "best four-lepton mass", "unit": "GeV", "range": [70, 250], "tooltip": "Four-lepton mass for the best ZZ-like candidate in the event."},
            "z1_mass": {"label": "Z1 candidate mass", "unit": "GeV", "range": [40, 120], "tooltip": "Mass of the pair closer to the known Z mass within a four-lepton candidate."},
            "z2_mass": {"label": "Z2 candidate mass", "unit": "GeV", "range": [0, 120], "tooltip": "Mass of the other pair in the ZZ candidate."},
            "score_looks_like_ZZ": {"label": "ZZ-like score", "unit": "GeV", "range": [0, 140], "tooltip": "A stored comparison quantity for teacher or technical views."},
            "pair_a_mass": {"label": "pair A mass", "unit": "GeV", "range": [0, 130], "tooltip": "Mass of one pair in a proposed split of four leptons."},
            "pair_b_mass": {"label": "pair B mass", "unit": "GeV", "range": [0, 130], "tooltip": "Mass of the other pair in a proposed split of four leptons."},
            "pair_a_energy": {"label": "pair A energy", "unit": "GeV", "range": [0, 250], "tooltip": "Total energy of pair A."},
            "pair_b_energy": {"label": "pair B energy", "unit": "GeV", "range": [0, 250], "tooltip": "Total energy of pair B."},
            "leading_lepton_pt": {"label": "leading lepton pT", "unit": "GeV", "range": [0, 250], "tooltip": "Largest transverse momentum among the four leptons."},
            "subleading_lepton_pt": {"label": "second lepton pT", "unit": "GeV", "range": [0, 180], "tooltip": "Second-largest transverse momentum among the four leptons."},
            "min_lepton_pt": {"label": "smallest lepton pT", "unit": "GeV", "range": [0, 100], "tooltip": "Smallest transverse momentum among the four leptons."},
            "max_abs_lepton_eta": {"label": "largest |eta|", "unit": "", "range": [0, 3], "tooltip": "How far forward the most forward of the four leptons is."},
            "max_lepton_iso": {"label": "largest lepton isolation", "unit": "", "range": [0, 2], "tooltip": "Largest surrounding activity relative to lepton pT; prompt isolated leptons tend to have small values."},
            "has_ossf_pairing": {"label": "has two OSSF pairs", "unit": "", "tooltip": "1 means the four leptons can be split into two opposite-sign, same-flavor pairs."},
            "pass_pt_hzz": {"label": "passes 20/10 GeV lepton pT thresholds", "unit": "", "tooltip": "At least one lepton has pT above 20 GeV and a second has pT above 10 GeV."},
            "pass_z_windows": {"label": "passes Z1/Z2 mass windows", "unit": "", "tooltip": "Z1 lies between 40 and 120 GeV and Z2 between 12 and 120 GeV."},
            "pass_isolation_035": {"label": "passes simplified isolation", "unit": "", "tooltip": "All four leptons have stored relative isolation below 0.35."},
        },
        "student_paths": [
            {
                "title": "Build a pair filter",
                "collection": "Dileptons",
                "start_variable": "mass",
                "suggested_filters": ["opposite charge", "same flavor"],
            },
            {
                "title": "Energy is not mass",
                "collection": "QuadLeptons",
                "compare_variables": ["energy", "momentum", "scalar_pt_sum", "mass"],
            },
            {
                "title": "The pairing puzzle",
                "collection": "PairingOptions",
                "compare_variables": ["pair_a_mass", "pair_b_mass", "score_looks_like_ZZ"],
            },
        ],
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a student-facing ZZ->4l masterclass ROOT file from NanoAOD.")
    p.add_argument("input", help="Input NanoAOD/NANOAODSIM ROOT file")
    p.add_argument("-o", "--output", default="zz4l_masterclass.root", help="Output reduced ROOT file")
    p.add_argument("--tree", default="Events", help="NanoAOD TTree name")
    p.add_argument("--sample-id", type=int, default=1)
    p.add_argument("--sample-name", default=None, help="Human-readable sample name. Defaults to the input filename stem.")
    p.add_argument("--muon-min-pt", type=float, default=5.0)
    p.add_argument("--muon-max-eta", type=float, default=2.4)
    p.add_argument("--electron-min-pt", type=float, default=7.0)
    p.add_argument("--electron-max-eta", type=float, default=2.5)
    p.add_argument("--electron-cutbased-min", type=int, default=1, help="Electron_cutBased minimum. 1=very loose, 2=loose, 3=medium, 4=tight.")
    p.add_argument("--step-size", default="100 MB")
    p.add_argument("--max-events", type=int, default=None, help="Optional maximum number of input events for quick tests")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    input_path = args.input
    output_path = Path(args.output)
    metadata_path = output_path.with_suffix(".metadata.json")

    with uproot.open(input_path) as fin:
        tree = fin[args.tree]
        available = set(tree.keys())

    required = [
        "Muon_pt", "Muon_eta", "Muon_phi", "Muon_charge",
        "Electron_pt", "Electron_eta", "Electron_phi", "Electron_charge",
    ]
    missing = [b for b in required if b not in available]
    if missing:
        raise RuntimeError(f"Missing required NanoAOD branches: {missing}")

    optional = [
        "run", "luminosityBlock", "event",
        "Muon_mass", "Muon_mediumId", "Muon_looseId", "Muon_pfRelIso04_all",
        "Electron_mass", "Electron_cutBased", "Electron_pfRelIso03_all",
    ]
    branches = required + [b for b in optional if b in available]

    if "Muon_mass" not in available:
        warnings.warn("Muon_mass not found; using 0.105658 GeV.")
    if "Electron_mass" not in available:
        warnings.warn("Electron_mass not found; using 0.000511 GeV.")
    if "Muon_mediumId" not in available and "Muon_looseId" not in available:
        warnings.warn("No Muon_mediumId or Muon_looseId found; using only kinematic muon cuts.")
    if "Electron_cutBased" not in available:
        warnings.warn("Electron_cutBased not found; using only kinematic electron cuts.")

    event_types = {
        "event_index": "int64", "run": "uint32", "luminosityBlock": "uint32", "event": "uint64",
        "sample_id": "int32", "n_lepton": "int32", "n_electron": "int32", "n_muon": "int32",
        "n_dilepton": "int32", "n_ossf_dilepton": "int32", "n_quad": "int32", "n_zz_candidate": "int32",
        "best_m4l": "float32", "best_z1_mass": "float32", "best_z2_mass": "float32", "best_channel": "int32",
    }
    lepton_types = {
        "event_index": "int64", "lepton_index": "int32", "source_index": "int32", "abs_pdgid": "int32", "pdgid": "int32", "charge": "int32",
        "pt": "float32", "eta": "float32", "phi": "float32", "mass": "float32", "energy": "float32",
        "px": "float32", "py": "float32", "pz": "float32", "momentum": "float32", "iso": "float32", "pass_id": "int32",
    }
    dilepton_types = {
        "event_index": "int64", "pair_index": "int32", "lep1_index": "int32", "lep2_index": "int32", "channel": "int32",
        "charge_sum": "int32", "is_os": "int32", "is_sf": "int32", "is_ossf": "int32",
        "mass": "float32", "mass_squared": "float32", "pt": "float32", "eta": "float32", "phi": "float32",
        "energy": "float32", "px": "float32", "py": "float32", "pz": "float32", "momentum": "float32",
        "energy_minus_momentum": "float32", "dr": "float32", "dphi": "float32", "opening_angle": "float32",
        "scalar_pt_sum": "float32", "abs_mass_minus_z": "float32",
    }
    quad_types = {
        "event_index": "int64", "quad_index": "int32", "lep1_index": "int32", "lep2_index": "int32", "lep3_index": "int32", "lep4_index": "int32",
        "channel": "int32", "charge_sum": "int32", "n_ossf_pairs": "int32", "is_zz_candidate": "int32", "is_best": "int32",
        "mass": "float32", "mass_squared": "float32", "pt": "float32", "eta": "float32", "phi": "float32",
        "energy": "float32", "px": "float32", "py": "float32", "pz": "float32", "momentum": "float32",
        "energy_minus_momentum": "float32", "scalar_pt_sum": "float32",
        "leading_lepton_pt": "float32", "subleading_lepton_pt": "float32", "min_lepton_pt": "float32",
        "max_abs_lepton_eta": "float32", "max_lepton_iso": "float32",
        "has_ossf_pairing": "int32", "pass_pt_hzz": "int32", "pass_z_windows": "int32", "pass_isolation_035": "int32",
        "best_pairing": "int32", "z1_mass": "float32", "z2_mass": "float32",
        "z1_lep1_index": "int32", "z1_lep2_index": "int32", "z2_lep1_index": "int32", "z2_lep2_index": "int32",
        "abs_z1_mass_minus_z": "float32", "abs_z2_mass_minus_z": "float32", "score_looks_like_ZZ": "float32",
    }
    pairing_types = {
        "event_index": "int64", "quad_index": "int32", "quad_is_best": "int32", "pairing_id": "int32",
        "quad_charge_sum": "int32",
        "pair_a_lep1_index": "int32", "pair_a_lep2_index": "int32", "pair_b_lep1_index": "int32", "pair_b_lep2_index": "int32",
        "pair_a_mass": "float32", "pair_a_mass_squared": "float32", "pair_a_pt": "float32", "pair_a_eta": "float32", "pair_a_phi": "float32",
        "pair_a_energy": "float32", "pair_a_px": "float32", "pair_a_py": "float32", "pair_a_pz": "float32", "pair_a_momentum": "float32",
        "pair_a_energy_minus_momentum": "float32", "pair_a_scalar_pt_sum": "float32", "pair_a_charge_sum": "int32",
        "pair_a_dr": "float32", "pair_a_dphi": "float32", "pair_a_opening_angle": "float32",
        "pair_b_mass": "float32", "pair_b_mass_squared": "float32", "pair_b_pt": "float32", "pair_b_eta": "float32", "pair_b_phi": "float32",
        "pair_b_energy": "float32", "pair_b_px": "float32", "pair_b_py": "float32", "pair_b_pz": "float32", "pair_b_momentum": "float32",
        "pair_b_energy_minus_momentum": "float32", "pair_b_scalar_pt_sum": "float32", "pair_b_charge_sum": "int32",
        "pair_b_dr": "float32", "pair_b_dphi": "float32", "pair_b_opening_angle": "float32",
        "z1_mass": "float32", "z2_mass": "float32",
        "pair_a_is_os": "int32", "pair_a_is_sf": "int32", "pair_a_is_ossf": "int32",
        "pair_b_is_os": "int32", "pair_b_is_sf": "int32", "pair_b_is_ossf": "int32",
        "both_pairs_ossf": "int32", "score_looks_like_ZZ": "float32",
    }

    global_event_index = 0
    entry_stop = args.max_events

    with uproot.recreate(output_path) as fout:
        fout.mktree("Events", event_types)
        fout.mktree("Leptons", lepton_types)
        fout.mktree("Dileptons", dilepton_types)
        fout.mktree("QuadLeptons", quad_types)
        fout.mktree("PairingOptions", pairing_types)

        for arrays in uproot.iterate(
            f"{input_path}:{args.tree}", branches, library="ak", step_size=args.step_size, entry_stop=entry_stop
        ):
            n_events = len(arrays["Muon_pt"])
            event_index = np.arange(global_event_index, global_event_index + n_events, dtype=np.int64)

            run = to_numpy(arrays["run"], np.uint32) if "run" in arrays.fields else np.zeros(n_events, dtype=np.uint32)
            lumi = to_numpy(arrays["luminosityBlock"], np.uint32) if "luminosityBlock" in arrays.fields else np.zeros(n_events, dtype=np.uint32)
            event = to_numpy(arrays["event"], np.uint64) if "event" in arrays.fields else event_index.astype(np.uint64)

            # Muons
            mu_mass = arrays["Muon_mass"] if "Muon_mass" in arrays.fields else jagged_const_like(arrays["Muon_pt"], 0.105658)
            mu_iso = arrays["Muon_pfRelIso04_all"] if "Muon_pfRelIso04_all" in arrays.fields else jagged_const_like(arrays["Muon_pt"], -1.0)
            if "Muon_mediumId" in arrays.fields:
                mu_pass_id = arrays["Muon_mediumId"]
            elif "Muon_looseId" in arrays.fields:
                mu_pass_id = arrays["Muon_looseId"]
            else:
                mu_pass_id = jagged_true_like(arrays["Muon_pt"])

            muons = ak.zip({
                "pt": arrays["Muon_pt"], "eta": arrays["Muon_eta"], "phi": arrays["Muon_phi"], "mass": mu_mass,
                "charge": arrays["Muon_charge"], "iso": mu_iso, "pass_id": mu_pass_id,
                "abs_pdgid": jagged_const_like(arrays["Muon_pt"], 13), "pdgid": -arrays["Muon_charge"] * 13,
                "source_index": ak.local_index(arrays["Muon_pt"]),
            }, with_name="Momentum4D")
            muons = muons[(muons.pt > args.muon_min_pt) & (abs(muons.eta) < args.muon_max_eta) & muons.pass_id]

            # Electrons
            el_mass = arrays["Electron_mass"] if "Electron_mass" in arrays.fields else jagged_const_like(arrays["Electron_pt"], 0.000511)
            el_iso = arrays["Electron_pfRelIso03_all"] if "Electron_pfRelIso03_all" in arrays.fields else jagged_const_like(arrays["Electron_pt"], -1.0)
            el_pass_id = arrays["Electron_cutBased"] >= args.electron_cutbased_min if "Electron_cutBased" in arrays.fields else jagged_true_like(arrays["Electron_pt"])

            electrons = ak.zip({
                "pt": arrays["Electron_pt"], "eta": arrays["Electron_eta"], "phi": arrays["Electron_phi"], "mass": el_mass,
                "charge": arrays["Electron_charge"], "iso": el_iso, "pass_id": el_pass_id,
                "abs_pdgid": jagged_const_like(arrays["Electron_pt"], 11), "pdgid": -arrays["Electron_charge"] * 11,
                "source_index": ak.local_index(arrays["Electron_pt"]),
            }, with_name="Momentum4D")
            electrons = electrons[(electrons.pt > args.electron_min_pt) & (abs(electrons.eta) < args.electron_max_eta) & electrons.pass_id]

            leptons = ak.concatenate([electrons, muons], axis=1)
            leptons = leptons[ak.argsort(leptons.pt, axis=1, ascending=False)]
            leptons = ak.with_field(leptons, ak.local_index(leptons.pt), "lepton_index")

            n_lepton = ak.num(leptons, axis=1)
            n_electron = ak.sum(abs(leptons.pdgid) == 11, axis=1)
            n_muon = ak.sum(abs(leptons.pdgid) == 13, axis=1)

            # Dileptons
            dileptons = ak.combinations(leptons, 2, fields=["l1", "l2"])
            d_p4 = dileptons.l1 + dileptons.l2
            d_is_os = dileptons.l1.charge * dileptons.l2.charge < 0
            d_is_sf = abs(dileptons.l1.pdgid) == abs(dileptons.l2.pdgid)
            d_is_ossf = d_is_os & d_is_sf
            d_n_e = one_if(abs(dileptons.l1.pdgid) == 11) + one_if(abs(dileptons.l2.pdgid) == 11)
            d_channel = ak.where(d_n_e == 2, 1, ak.where(d_n_e == 0, 2, 3))  # 1=ee, 2=mumu, 3=emu
            d_mass = d_p4.mass
            d_momentum = momentum_mag(d_p4)

            # Quadleptons
            quads = ak.combinations(leptons, 4, fields=["l1", "l2", "l3", "l4"])
            q_p4 = quads.l1 + quads.l2 + quads.l3 + quads.l4
            q_mass = q_p4.mass
            q_momentum = momentum_mag(q_p4)
            q_charge_sum = quads.l1.charge + quads.l2.charge + quads.l3.charge + quads.l4.charge
            q_n_e = (one_if(abs(quads.l1.pdgid) == 11) + one_if(abs(quads.l2.pdgid) == 11) + one_if(abs(quads.l3.pdgid) == 11) + one_if(abs(quads.l4.pdgid) == 11))
            q_channel = ak.where(q_n_e == 4, 1, ak.where(q_n_e == 0, 2, ak.where(q_n_e == 2, 3, 0)))  # 1=4e, 2=4mu, 3=2e2mu, 0=other

            # Six possible two-lepton objects inside a four-lepton candidate.
            p12 = quads.l1 + quads.l2
            p13 = quads.l1 + quads.l3
            p14 = quads.l1 + quads.l4
            p23 = quads.l2 + quads.l3
            p24 = quads.l2 + quads.l4
            p34 = quads.l3 + quads.l4
            m12 = p12.mass
            m13 = p13.mass
            m14 = p14.mass
            m23 = p23.mass
            m24 = p24.mass
            m34 = p34.mass

            os12 = quads.l1.charge * quads.l2.charge < 0
            os13 = quads.l1.charge * quads.l3.charge < 0
            os14 = quads.l1.charge * quads.l4.charge < 0
            os23 = quads.l2.charge * quads.l3.charge < 0
            os24 = quads.l2.charge * quads.l4.charge < 0
            os34 = quads.l3.charge * quads.l4.charge < 0
            sf12 = abs(quads.l1.pdgid) == abs(quads.l2.pdgid)
            sf13 = abs(quads.l1.pdgid) == abs(quads.l3.pdgid)
            sf14 = abs(quads.l1.pdgid) == abs(quads.l4.pdgid)
            sf23 = abs(quads.l2.pdgid) == abs(quads.l3.pdgid)
            sf24 = abs(quads.l2.pdgid) == abs(quads.l4.pdgid)
            sf34 = abs(quads.l3.pdgid) == abs(quads.l4.pdgid)
            ossf12, ossf13, ossf14 = os12 & sf12, os13 & sf13, os14 & sf14
            ossf23, ossf24, ossf34 = os23 & sf23, os24 & sf24, os34 & sf34
            n_ossf_pairs = one_if(ossf12) + one_if(ossf13) + one_if(ossf14) + one_if(ossf23) + one_if(ossf24) + one_if(ossf34)

            # Three possible pairings of four objects.
            # 0=(12)(34), 1=(13)(24), 2=(14)(23)
            p0_ok = ossf12 & ossf34
            p1_ok = ossf13 & ossf24
            p2_ok = ossf14 & ossf23
            p0 = choose_pairing_z_order(m12, quads.l1.lepton_index, quads.l2.lepton_index, m34, quads.l3.lepton_index, quads.l4.lepton_index)
            p1 = choose_pairing_z_order(m13, quads.l1.lepton_index, quads.l3.lepton_index, m24, quads.l2.lepton_index, quads.l4.lepton_index)
            p2 = choose_pairing_z_order(m14, quads.l1.lepton_index, quads.l4.lepton_index, m23, quads.l2.lepton_index, quads.l3.lepton_index)
            p0_score = ak.where(p0_ok, abs(p0[0] - Z_MASS_GEV) + 0.25 * abs(p0[1] - Z_MASS_GEV), 1.0e9)
            p1_score = ak.where(p1_ok, abs(p1[0] - Z_MASS_GEV) + 0.25 * abs(p1[1] - Z_MASS_GEV), 1.0e9)
            p2_score = ak.where(p2_ok, abs(p2[0] - Z_MASS_GEV) + 0.25 * abs(p2[1] - Z_MASS_GEV), 1.0e9)
            best_pairing = ak.where((p0_score <= p1_score) & (p0_score <= p2_score), 0, ak.where(p1_score <= p2_score, 1, 2))

            def choose(a: ak.Array, b: ak.Array, c: ak.Array) -> ak.Array:
                return ak.where(best_pairing == 0, a, ak.where(best_pairing == 1, b, c))

            z1_mass = choose(p0[0], p1[0], p2[0])
            z2_mass = choose(p0[1], p1[1], p2[1])
            z1_l1_index = choose(p0[2], p1[2], p2[2])
            z1_l2_index = choose(p0[3], p1[3], p2[3])
            z2_l1_index = choose(p0[4], p1[4], p2[4])
            z2_l2_index = choose(p0[5], p1[5], p2[5])
            score_looks_like_ZZ = choose(p0_score, p1_score, p2_score)
            has_ossf_pairing = p0_ok | p1_ok | p2_ok
            is_zz_candidate = (score_looks_like_ZZ < 1.0e8) & (q_charge_sum == 0)
            q_leading_pt = quads.l1.pt
            q_subleading_pt = quads.l2.pt
            q_min_lepton_pt = quads.l4.pt
            q_max_abs_eta = ak.where(
                abs(quads.l1.eta) > abs(quads.l2.eta), abs(quads.l1.eta), abs(quads.l2.eta)
            )
            q_max_abs_eta = ak.where(q_max_abs_eta > abs(quads.l3.eta), q_max_abs_eta, abs(quads.l3.eta))
            q_max_abs_eta = ak.where(q_max_abs_eta > abs(quads.l4.eta), q_max_abs_eta, abs(quads.l4.eta))
            q_max_iso = ak.where(quads.l1.iso > quads.l2.iso, quads.l1.iso, quads.l2.iso)
            q_max_iso = ak.where(q_max_iso > quads.l3.iso, q_max_iso, quads.l3.iso)
            q_max_iso = ak.where(q_max_iso > quads.l4.iso, q_max_iso, quads.l4.iso)
            q_pass_pt_hzz = (q_leading_pt > 20.0) & (q_subleading_pt > 10.0)
            q_pass_z_windows = (z1_mass > 40.0) & (z1_mass < 120.0) & (z2_mass > 12.0) & (z2_mass < 120.0)
            q_pass_isolation = (q_max_iso < 0.35) | (q_max_iso < 0.0)

            # Best quad per event: the valid candidate with lowest score.
            q_event_index_flat = flat_event_index(q_mass, event_index)
            q_mass_flat = flat_numpy(q_mass, np.float32)
            q_z1_flat = flat_numpy(z1_mass, np.float32)
            q_z2_flat = flat_numpy(z2_mass, np.float32)
            q_channel_flat = flat_numpy(q_channel, np.int32)
            q_is_zz_flat = flat_numpy(is_zz_candidate, np.int32)
            q_score_flat = flat_numpy(score_looks_like_ZZ, np.float32)
            q_score_for_best = np.where(q_is_zz_flat == 1, q_score_flat, np.inf)

            best_m4l = np.full(n_events, np.nan, dtype=np.float32)
            best_z1 = np.full(n_events, np.nan, dtype=np.float32)
            best_z2 = np.full(n_events, np.nan, dtype=np.float32)
            best_channel = np.zeros(n_events, dtype=np.int32)
            best_score = np.full(n_events, np.inf, dtype=np.float64)
            best_flat_index = np.full(n_events, -1, dtype=np.int64)

            for i in np.argsort(q_score_for_best):
                if not np.isfinite(q_score_for_best[i]):
                    continue
                local_event = int(q_event_index_flat[i] - global_event_index)
                if q_score_for_best[i] < best_score[local_event]:
                    best_score[local_event] = q_score_for_best[i]
                    best_m4l[local_event] = q_mass_flat[i]
                    best_z1[local_event] = q_z1_flat[i]
                    best_z2[local_event] = q_z2_flat[i]
                    best_channel[local_event] = q_channel_flat[i]
                    best_flat_index[local_event] = i

            q_is_best_flat = np.zeros(len(q_mass_flat), dtype=np.int32)
            valid_best = best_flat_index >= 0
            q_is_best_flat[best_flat_index[valid_best]] = 1

            # Events
            fout["Events"].extend({
                "event_index": event_index, "run": run, "luminosityBlock": lumi, "event": event,
                "sample_id": np.full(n_events, args.sample_id, dtype=np.int32),
                "n_lepton": to_numpy(n_lepton, np.int32),
                "n_electron": to_numpy(n_electron, np.int32),
                "n_muon": to_numpy(n_muon, np.int32),
                "n_dilepton": to_numpy(ak.num(d_mass, axis=1), np.int32),
                "n_ossf_dilepton": to_numpy(ak.sum(d_is_ossf, axis=1), np.int32),
                "n_quad": to_numpy(ak.num(q_mass, axis=1), np.int32),
                "n_zz_candidate": to_numpy(ak.sum(is_zz_candidate, axis=1), np.int32),
                "best_m4l": best_m4l, "best_z1_mass": best_z1, "best_z2_mass": best_z2, "best_channel": best_channel,
            })

            # Leptons
            write_if_nonempty(fout["Leptons"], {
                "event_index": flat_event_index(leptons.pt, event_index),
                "lepton_index": flat_numpy(leptons.lepton_index, np.int32),
                "source_index": flat_numpy(leptons.source_index, np.int32),
                "abs_pdgid": flat_numpy(leptons.abs_pdgid, np.int32),
                "pdgid": flat_numpy(leptons.pdgid, np.int32),
                "charge": flat_numpy(leptons.charge, np.int32),
                "pt": flat_numpy(leptons.pt, np.float32),
                "eta": flat_numpy(leptons.eta, np.float32),
                "phi": flat_numpy(leptons.phi, np.float32),
                "mass": flat_numpy(leptons.mass, np.float32),
                "energy": flat_numpy(leptons.energy, np.float32),
                "px": flat_numpy(leptons.px, np.float32),
                "py": flat_numpy(leptons.py, np.float32),
                "pz": flat_numpy(leptons.pz, np.float32),
                "momentum": flat_numpy(momentum_mag(leptons), np.float32),
                "iso": flat_numpy(leptons.iso, np.float32),
                "pass_id": flat_numpy(leptons.pass_id, np.int32),
            })

            # Dileptons
            write_if_nonempty(fout["Dileptons"], {
                "event_index": flat_event_index(d_mass, event_index),
                "pair_index": flat_numpy(ak.local_index(d_mass), np.int32),
                "lep1_index": flat_numpy(dileptons.l1.lepton_index, np.int32),
                "lep2_index": flat_numpy(dileptons.l2.lepton_index, np.int32),
                "channel": flat_numpy(d_channel, np.int32),
                "charge_sum": flat_numpy(dileptons.l1.charge + dileptons.l2.charge, np.int32),
                "is_os": flat_numpy(d_is_os, np.int32),
                "is_sf": flat_numpy(d_is_sf, np.int32),
                "is_ossf": flat_numpy(d_is_ossf, np.int32),
                "mass": flat_numpy(d_p4.mass, np.float32),
                "mass_squared": flat_numpy(mass_squared(d_p4), np.float32),
                "pt": flat_numpy(d_p4.pt, np.float32),
                "eta": flat_numpy(d_p4.eta, np.float32),
                "phi": flat_numpy(d_p4.phi, np.float32),
                "energy": flat_numpy(d_p4.energy, np.float32),
                "px": flat_numpy(d_p4.px, np.float32),
                "py": flat_numpy(d_p4.py, np.float32),
                "pz": flat_numpy(d_p4.pz, np.float32),
                "momentum": flat_numpy(d_momentum, np.float32),
                "energy_minus_momentum": flat_numpy(d_p4.energy - d_momentum, np.float32),
                "dr": flat_numpy(dileptons.l1.deltaR(dileptons.l2), np.float32),
                "dphi": flat_numpy(dileptons.l1.deltaphi(dileptons.l2), np.float32),
                "opening_angle": flat_numpy(opening_angle(dileptons.l1, dileptons.l2), np.float32),
                "scalar_pt_sum": flat_numpy(dileptons.l1.pt + dileptons.l2.pt, np.float32),
                "abs_mass_minus_z": flat_numpy(abs(d_p4.mass - Z_MASS_GEV), np.float32),
            })

            # QuadLeptons
            write_if_nonempty(fout["QuadLeptons"], {
                "event_index": q_event_index_flat,
                "quad_index": flat_numpy(ak.local_index(q_mass), np.int32),
                "lep1_index": flat_numpy(quads.l1.lepton_index, np.int32),
                "lep2_index": flat_numpy(quads.l2.lepton_index, np.int32),
                "lep3_index": flat_numpy(quads.l3.lepton_index, np.int32),
                "lep4_index": flat_numpy(quads.l4.lepton_index, np.int32),
                "channel": q_channel_flat,
                "charge_sum": flat_numpy(q_charge_sum, np.int32),
                "n_ossf_pairs": flat_numpy(n_ossf_pairs, np.int32),
                "is_zz_candidate": q_is_zz_flat,
                "is_best": q_is_best_flat,
                "mass": q_mass_flat,
                "mass_squared": flat_numpy(mass_squared(q_p4), np.float32),
                "pt": flat_numpy(q_p4.pt, np.float32),
                "eta": flat_numpy(q_p4.eta, np.float32),
                "phi": flat_numpy(q_p4.phi, np.float32),
                "energy": flat_numpy(q_p4.energy, np.float32),
                "px": flat_numpy(q_p4.px, np.float32),
                "py": flat_numpy(q_p4.py, np.float32),
                "pz": flat_numpy(q_p4.pz, np.float32),
                "momentum": flat_numpy(q_momentum, np.float32),
                "energy_minus_momentum": flat_numpy(q_p4.energy - q_momentum, np.float32),
                "scalar_pt_sum": flat_numpy(quads.l1.pt + quads.l2.pt + quads.l3.pt + quads.l4.pt, np.float32),
                "leading_lepton_pt": flat_numpy(q_leading_pt, np.float32),
                "subleading_lepton_pt": flat_numpy(q_subleading_pt, np.float32),
                "min_lepton_pt": flat_numpy(q_min_lepton_pt, np.float32),
                "max_abs_lepton_eta": flat_numpy(q_max_abs_eta, np.float32),
                "max_lepton_iso": flat_numpy(q_max_iso, np.float32),
                "has_ossf_pairing": flat_numpy(has_ossf_pairing, np.int32),
                "pass_pt_hzz": flat_numpy(q_pass_pt_hzz, np.int32),
                "pass_z_windows": flat_numpy(q_pass_z_windows, np.int32),
                "pass_isolation_035": flat_numpy(q_pass_isolation, np.int32),
                "best_pairing": flat_numpy(best_pairing, np.int32),
                "z1_mass": q_z1_flat,
                "z2_mass": q_z2_flat,
                "z1_lep1_index": flat_numpy(z1_l1_index, np.int32),
                "z1_lep2_index": flat_numpy(z1_l2_index, np.int32),
                "z2_lep1_index": flat_numpy(z2_l1_index, np.int32),
                "z2_lep2_index": flat_numpy(z2_l2_index, np.int32),
                "abs_z1_mass_minus_z": flat_numpy(abs(z1_mass - Z_MASS_GEV), np.float32),
                "abs_z2_mass_minus_z": flat_numpy(abs(z2_mass - Z_MASS_GEV), np.float32),
                "score_looks_like_ZZ": q_score_flat,
            })

            # PairingOptions: flatten and concatenate 3 options per quad.
            qidx_flat = flat_numpy(ak.local_index(q_mass), np.int32)
            def cat3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
                return np.concatenate([a, b, c]) if len(a) or len(b) or len(c) else np.array([], dtype=a.dtype)

            def f32(x: ak.Array) -> np.ndarray:
                return flat_numpy(x, np.float32)

            def i32(x: ak.Array) -> np.ndarray:
                return flat_numpy(x, np.int32)

            def cat_pair_field(a: ak.Array, b: ak.Array, c: ak.Array, field: str) -> np.ndarray:
                return cat3(f32(getattr(a, field)), f32(getattr(b, field)), f32(getattr(c, field)))

            def cat_pair_extra(a: ak.Array, b: ak.Array, c: ak.Array, expr) -> np.ndarray:
                return cat3(f32(expr(a)), f32(expr(b)), f32(expr(c)))

            p0_z1 = f32(p0[0]); p0_z2 = f32(p0[1])
            p1_z1 = f32(p1[0]); p1_z2 = f32(p1[1])
            p2_z1 = f32(p2[0]); p2_z2 = f32(p2[1])

            write_if_nonempty(fout["PairingOptions"], {
                "event_index": np.tile(q_event_index_flat, 3),
                "quad_index": np.tile(qidx_flat, 3),
                "quad_is_best": np.tile(q_is_best_flat, 3),
                "pairing_id": np.concatenate([np.zeros_like(qidx_flat), np.ones_like(qidx_flat), np.full_like(qidx_flat, 2)]),
                "quad_charge_sum": cat3(i32(q_charge_sum), i32(q_charge_sum), i32(q_charge_sum)),
                "pair_a_lep1_index": cat3(i32(quads.l1.lepton_index), i32(quads.l1.lepton_index), i32(quads.l1.lepton_index)),
                "pair_a_lep2_index": cat3(i32(quads.l2.lepton_index), i32(quads.l3.lepton_index), i32(quads.l4.lepton_index)),
                "pair_b_lep1_index": cat3(i32(quads.l3.lepton_index), i32(quads.l2.lepton_index), i32(quads.l2.lepton_index)),
                "pair_b_lep2_index": cat3(i32(quads.l4.lepton_index), i32(quads.l4.lepton_index), i32(quads.l3.lepton_index)),

                "pair_a_mass": cat_pair_field(p12, p13, p14, "mass"),
                "pair_a_mass_squared": cat_pair_extra(p12, p13, p14, mass_squared),
                "pair_a_pt": cat_pair_field(p12, p13, p14, "pt"),
                "pair_a_eta": cat_pair_field(p12, p13, p14, "eta"),
                "pair_a_phi": cat_pair_field(p12, p13, p14, "phi"),
                "pair_a_energy": cat_pair_field(p12, p13, p14, "energy"),
                "pair_a_px": cat_pair_field(p12, p13, p14, "px"),
                "pair_a_py": cat_pair_field(p12, p13, p14, "py"),
                "pair_a_pz": cat_pair_field(p12, p13, p14, "pz"),
                "pair_a_momentum": cat_pair_extra(p12, p13, p14, momentum_mag),
                "pair_a_energy_minus_momentum": cat3(f32(p12.energy - momentum_mag(p12)), f32(p13.energy - momentum_mag(p13)), f32(p14.energy - momentum_mag(p14))),
                "pair_a_scalar_pt_sum": cat3(f32(quads.l1.pt + quads.l2.pt), f32(quads.l1.pt + quads.l3.pt), f32(quads.l1.pt + quads.l4.pt)),
                "pair_a_charge_sum": cat3(i32(quads.l1.charge + quads.l2.charge), i32(quads.l1.charge + quads.l3.charge), i32(quads.l1.charge + quads.l4.charge)),
                "pair_a_dr": cat3(f32(quads.l1.deltaR(quads.l2)), f32(quads.l1.deltaR(quads.l3)), f32(quads.l1.deltaR(quads.l4))),
                "pair_a_dphi": cat3(f32(quads.l1.deltaphi(quads.l2)), f32(quads.l1.deltaphi(quads.l3)), f32(quads.l1.deltaphi(quads.l4))),
                "pair_a_opening_angle": cat3(f32(opening_angle(quads.l1, quads.l2)), f32(opening_angle(quads.l1, quads.l3)), f32(opening_angle(quads.l1, quads.l4))),

                "pair_b_mass": cat_pair_field(p34, p24, p23, "mass"),
                "pair_b_mass_squared": cat_pair_extra(p34, p24, p23, mass_squared),
                "pair_b_pt": cat_pair_field(p34, p24, p23, "pt"),
                "pair_b_eta": cat_pair_field(p34, p24, p23, "eta"),
                "pair_b_phi": cat_pair_field(p34, p24, p23, "phi"),
                "pair_b_energy": cat_pair_field(p34, p24, p23, "energy"),
                "pair_b_px": cat_pair_field(p34, p24, p23, "px"),
                "pair_b_py": cat_pair_field(p34, p24, p23, "py"),
                "pair_b_pz": cat_pair_field(p34, p24, p23, "pz"),
                "pair_b_momentum": cat_pair_extra(p34, p24, p23, momentum_mag),
                "pair_b_energy_minus_momentum": cat3(f32(p34.energy - momentum_mag(p34)), f32(p24.energy - momentum_mag(p24)), f32(p23.energy - momentum_mag(p23))),
                "pair_b_scalar_pt_sum": cat3(f32(quads.l3.pt + quads.l4.pt), f32(quads.l2.pt + quads.l4.pt), f32(quads.l2.pt + quads.l3.pt)),
                "pair_b_charge_sum": cat3(i32(quads.l3.charge + quads.l4.charge), i32(quads.l2.charge + quads.l4.charge), i32(quads.l2.charge + quads.l3.charge)),
                "pair_b_dr": cat3(f32(quads.l3.deltaR(quads.l4)), f32(quads.l2.deltaR(quads.l4)), f32(quads.l2.deltaR(quads.l3))),
                "pair_b_dphi": cat3(f32(quads.l3.deltaphi(quads.l4)), f32(quads.l2.deltaphi(quads.l4)), f32(quads.l2.deltaphi(quads.l3))),
                "pair_b_opening_angle": cat3(f32(opening_angle(quads.l3, quads.l4)), f32(opening_angle(quads.l2, quads.l4)), f32(opening_angle(quads.l2, quads.l3))),

                "z1_mass": cat3(p0_z1, p1_z1, p2_z1),
                "z2_mass": cat3(p0_z2, p1_z2, p2_z2),
                "pair_a_is_os": cat3(i32(os12), i32(os13), i32(os14)),
                "pair_a_is_sf": cat3(i32(sf12), i32(sf13), i32(sf14)),
                "pair_a_is_ossf": cat3(i32(ossf12), i32(ossf13), i32(ossf14)),
                "pair_b_is_os": cat3(i32(os34), i32(os24), i32(os23)),
                "pair_b_is_sf": cat3(i32(sf34), i32(sf24), i32(sf23)),
                "pair_b_is_ossf": cat3(i32(ossf34), i32(ossf24), i32(ossf23)),
                "both_pairs_ossf": cat3(i32(p0_ok), i32(p1_ok), i32(p2_ok)),
                "score_looks_like_ZZ": cat3(f32(p0_score), f32(p1_score), f32(p2_score)),
            })

            global_event_index += n_events

    metadata = make_metadata(output_path, args.sample_name or Path(input_path).stem, args.sample_id, input_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {output_path}")
    print(f"Wrote {metadata_path}")
    print(f"Processed {global_event_index:,} input events")


if __name__ == "__main__":
    main()
