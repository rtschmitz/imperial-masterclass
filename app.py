#!/usr/bin/env python3
"""
FastAPI backend for the ZZ->4l masterclass combination-builder tool.

Run after producing zz4l_masterclass.root:

  export MASTERCLASS_ROOT=../zz4l_masterclass.root
  uvicorn app:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000
"""

from __future__ import annotations

import ast
import json
import math
import os
import zlib
from pathlib import Path
from typing import Any, Literal

import numpy as np
import uproot
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
APP_VERSION = "3.2.8"

DEFAULT_ROOT_PATH = APP_DIR / "data" / "zz4l_masterclass.root"
if not DEFAULT_ROOT_PATH.exists():
    DEFAULT_ROOT_PATH = APP_DIR.parent / "zz4l_masterclass.root"
ROOT_PATH = Path(os.environ.get("MASTERCLASS_ROOT", DEFAULT_ROOT_PATH)).resolve()
METADATA_PATH = Path(os.environ.get("MASTERCLASS_METADATA", ROOT_PATH.with_suffix(".metadata.json"))).resolve()
EVENT_DISPLAY_URL_TEMPLATE = os.environ.get("EVENT_DISPLAY_URL_TEMPLATE", "")
EVENT_DISPLAY_URL = os.environ.get("EVENT_DISPLAY_URL", "https://ispy-webgl-masterclass.web.cern.ch/")

COLLECTION_ORDER = ["Events", "Leptons", "Dileptons", "QuadLeptons", "PairingOptions"]

PAIRING_RULES = {
    "closest_distance",
    "furthest_distance",
    "closest_formula",
    "furthest_formula",
    "same_type_same_charge",
    "same_type_different_charge",
    "different_type_same_charge",
    "different_type_different_charge",
    "random",
}

# Accept requests from an older cached frontend long enough for the browser to
# receive the current no-cache HTML and JavaScript.
LEGACY_PAIRING_RULES = {
    "closest_target": "closest_distance",
    "smallest_difference": "closest_formula",
    "largest_difference": "furthest_formula",
    "first": "closest_distance",
    "largest_sum": "closest_distance",
    "smallest_sum": "closest_distance",
    "largest_value": "closest_distance",
    "smallest_value": "closest_distance",
}

LEGACY_REQUIREMENT_IDS = {
    "pair_min_scalar_pt_sum": "pair_min_pt_total",
    "quad_min_scalar_pt_sum": "quad_min_pt_total",
}

REQUIREMENT_CATALOG: list[dict[str, Any]] = [
    {
        "id": "pair_opposite_charge",
        "label": "Both pairs have different charges",
        "description": "Requires a positive and a negative lepton in each selected pair.",
        "modes": ["pairs"],
        "comparison": "required",
        "default": 1,
        "unit": "",
        "fields": ["pair_a_is_os", "pair_b_is_os"],
    },
    {
        "id": "pair_same_type",
        "label": "Both pairs contain the same lepton type",
        "description": "Requires ee or μμ rather than eμ in each selected pair.",
        "modes": ["pairs"],
        "comparison": "required",
        "default": 1,
        "unit": "",
        "fields": ["pair_a_is_sf", "pair_b_is_sf"],
    },
    {
        "id": "pair_same_type_different_charge",
        "label": "Same type, different charge",
        "description": "Requires both selected pairs to be ee or μμ with one positive and one negative lepton.",
        "modes": ["pairs"],
        "comparison": "required",
        "default": 1,
        "unit": "",
        "fields": ["pair_a_is_sf", "pair_b_is_sf", "pair_a_is_os", "pair_b_is_os"],
        "expected": [1, 1, 1, 1],
    },
    {
        "id": "pair_same_type_same_charge",
        "label": "Same type, same charge",
        "description": "Requires both selected pairs to be ee or μμ with matching charges.",
        "modes": ["pairs"],
        "comparison": "required",
        "default": 1,
        "unit": "",
        "fields": ["pair_a_is_sf", "pair_b_is_sf", "pair_a_is_ss", "pair_b_is_ss"],
        "expected": [1, 1, 1, 1],
    },
    {
        "id": "pair_different_type_different_charge",
        "label": "Different type, different charge",
        "description": "Requires both selected pairs to contain one electron and one muon with different charges.",
        "modes": ["pairs"],
        "comparison": "required",
        "default": 1,
        "unit": "",
        "fields": ["pair_a_is_sf", "pair_b_is_sf", "pair_a_is_os", "pair_b_is_os"],
        "expected": [0, 0, 1, 1],
    },
    {
        "id": "pair_different_type_same_charge",
        "label": "Different type, same charge",
        "description": "Requires both selected pairs to contain one electron and one muon with matching charges.",
        "modes": ["pairs"],
        "comparison": "required",
        "default": 1,
        "unit": "",
        "fields": ["pair_a_is_sf", "pair_b_is_sf", "pair_a_is_ss", "pair_b_is_ss"],
        "expected": [0, 0, 1, 1],
    },
    {
        "id": "pair_min_pt",
        "label": "Each pair pT",
        "description": "Keeps events where both selected pairs exceed the threshold.",
        "modes": ["pairs"],
        "comparison": ">",
        "default": 20,
        "unit": "GeV",
        "fields": ["pair_a_pt", "pair_b_pt"],
    },
    {
        "id": "pair_min_energy",
        "label": "Each pair energy",
        "description": "Keeps events where both selected pairs exceed the threshold.",
        "modes": ["pairs"],
        "comparison": ">",
        "default": 35,
        "unit": "GeV",
        "fields": ["pair_a_energy", "pair_b_energy"],
    },
    {
        "id": "pair_min_pt_total",
        "label": "Each pair scalar pT",
        "description": "Adds the two lepton pT values without directional cancellation.",
        "modes": ["pairs"],
        "comparison": ">",
        "default": 30,
        "unit": "GeV",
        "fields": ["pair_a_pt_scalar", "pair_b_pt_scalar"],
    },
    {
        "id": "pair_min_momentum",
        "label": "Each pair momentum",
        "description": "Keeps events where both selected pairs exceed the momentum threshold.",
        "modes": ["pairs"],
        "comparison": ">",
        "default": 25,
        "unit": "GeV",
        "fields": ["pair_a_momentum", "pair_b_momentum"],
    },
    {
        "id": "pair_min_mass",
        "label": "Each pair mass — minimum",
        "description": "Keeps events where both selected pair masses are above the threshold.",
        "modes": ["pairs"],
        "comparison": ">",
        "default": 12,
        "unit": "GeV",
        "fields": ["pair_a_mass", "pair_b_mass"],
    },
    {
        "id": "pair_max_mass",
        "label": "Each pair mass — maximum",
        "description": "Keeps events where both selected pair masses are below the threshold.",
        "modes": ["pairs"],
        "comparison": "<",
        "default": 120,
        "unit": "GeV",
        "fields": ["pair_a_mass", "pair_b_mass"],
    },
    {
        "id": "pair_min_dr",
        "label": "Separation of each pair (ΔR)",
        "description": "Removes pairs whose leptons are very close in detector angle.",
        "modes": ["pairs"],
        "comparison": ">",
        "default": 0.4,
        "unit": "",
        "fields": ["pair_a_dr", "pair_b_dr"],
    },
    {
        "id": "pair_max_dr",
        "label": "Separation of each pair (ΔR) — maximum",
        "description": "Keeps pairs whose leptons are not farther apart than the threshold.",
        "modes": ["pairs"],
        "comparison": "<",
        "default": 3.5,
        "unit": "",
        "fields": ["pair_a_dr", "pair_b_dr"],
    },
    {
        "id": "pair_min_opening_angle",
        "label": "Opening angle of each pair — minimum",
        "description": "Keeps events where both pair opening angles exceed the threshold.",
        "modes": ["pairs"],
        "comparison": ">",
        "default": 0.5,
        "unit": "rad",
        "fields": ["pair_a_opening_angle", "pair_b_opening_angle"],
    },
    {
        "id": "pair_max_opening_angle",
        "label": "Opening angle of each pair — maximum",
        "description": "Keeps events where both pair opening angles are below the threshold.",
        "modes": ["pairs"],
        "comparison": "<",
        "default": 3.0,
        "unit": "rad",
        "fields": ["pair_a_opening_angle", "pair_b_opening_angle"],
    },
    {
        "id": "quad_charge_zero",
        "label": "Absolute four-lepton charge",
        "description": "A neutral parent should have a charge sum close to zero.",
        "modes": ["four"],
        "comparison": "≤",
        "default": 0,
        "unit": "",
        "fields": ["charge_sum"],
    },
    {
        "id": "leading_lepton_pt",
        "label": "Leading lepton pT",
        "description": "The H→ZZ→4ℓ selection requires one lepton above 20 GeV.",
        "modes": ["four"],
        "comparison": ">",
        "default": 20,
        "unit": "GeV",
        "fields": ["leading_lepton_pt"],
    },
    {
        "id": "subleading_lepton_pt",
        "label": "Second-highest lepton pT",
        "description": "The H→ZZ→4ℓ selection requires another lepton above 10 GeV.",
        "modes": ["four"],
        "comparison": ">",
        "default": 10,
        "unit": "GeV",
        "fields": ["subleading_lepton_pt"],
    },
    {
        "id": "minimum_lepton_pt",
        "label": "Lowest lepton pT",
        "description": "Requires every lepton in the four-lepton candidate to exceed the threshold.",
        "modes": ["four"],
        "comparison": ">",
        "default": 5,
        "unit": "GeV",
        "fields": ["min_lepton_pt"],
    },
    {
        "id": "max_lepton_iso",
        "label": "Largest lepton isolation",
        "description": "Prompt leptons are usually isolated from nearby activity.",
        "modes": ["four"],
        "comparison": "<",
        "default": 0.35,
        "unit": "",
        "fields": ["max_lepton_iso"],
    },
    {
        "id": "max_abs_eta",
        "label": "Largest lepton |η|",
        "description": "Restricts all four leptons to the detector acceptance.",
        "modes": ["four"],
        "comparison": "<",
        "default": 2.4,
        "unit": "",
        "fields": ["max_abs_lepton_eta"],
    },
    {
        "id": "z1_mass_min",
        "label": "Higher-mass Z candidate",
        "description": "Sets the lower edge of the Z1 candidate mass window.",
        "modes": ["four"],
        "comparison": ">",
        "default": 40,
        "unit": "GeV",
        "fields": ["z1_mass"],
    },
    {
        "id": "z2_mass_min",
        "label": "Lower-mass Z candidate",
        "description": "Sets the lower edge of the Z2 candidate mass window.",
        "modes": ["four"],
        "comparison": ">",
        "default": 12,
        "unit": "GeV",
        "fields": ["z2_mass"],
    },
    {
        "id": "z1_mass_max",
        "label": "Higher-mass Z candidate",
        "description": "Sets the upper edge of the Z1 candidate mass window.",
        "modes": ["four"],
        "comparison": "<",
        "default": 120,
        "unit": "GeV",
        "fields": ["z1_mass"],
    },
    {
        "id": "z2_mass_max",
        "label": "Lower-mass Z candidate",
        "description": "Sets the upper edge of the Z2 candidate mass window.",
        "modes": ["four"],
        "comparison": "<",
        "default": 120,
        "unit": "GeV",
        "fields": ["z2_mass"],
    },
    {
        "id": "z1_mass_distance",
        "label": "Z1 distance from the Z mass",
        "description": "Limits how far the higher-mass candidate is from the known Z mass.",
        "modes": ["four"],
        "comparison": "<",
        "default": 20,
        "unit": "GeV",
        "fields": ["abs_z1_mass_minus_z"],
    },
    {
        "id": "z2_mass_distance",
        "label": "Z2 distance from the Z mass",
        "description": "Limits how far the lower-mass candidate is from the known Z mass.",
        "modes": ["four"],
        "comparison": "<",
        "default": 65,
        "unit": "GeV",
        "fields": ["abs_z2_mass_minus_z"],
    },
    {
        "id": "quad_min_pt",
        "label": "Four-lepton pT",
        "description": "An exploratory threshold on the combined transverse momentum.",
        "modes": ["four"],
        "comparison": ">",
        "default": 20,
        "unit": "GeV",
        "fields": ["pt"],
    },
    {
        "id": "quad_min_energy",
        "label": "Four-lepton energy",
        "description": "An exploratory threshold on the combined energy.",
        "modes": ["four"],
        "comparison": ">",
        "default": 120,
        "unit": "GeV",
        "fields": ["energy"],
    },
    {
        "id": "quad_min_pt_total",
        "label": "Four-lepton scalar pT",
        "description": "Adds all four lepton pT values without directional cancellation.",
        "modes": ["four"],
        "comparison": ">",
        "default": 60,
        "unit": "GeV",
        "fields": ["pt_scalar"],
    },
    {
        "id": "quad_min_momentum",
        "label": "Four-lepton momentum",
        "description": "An exploratory threshold on the combined three-dimensional momentum.",
        "modes": ["four"],
        "comparison": ">",
        "default": 30,
        "unit": "GeV",
        "fields": ["momentum"],
    },
    {
        "id": "quad_min_mass",
        "label": "Four-lepton mass — minimum",
        "description": "Keeps candidates above the selected four-lepton mass.",
        "modes": ["four"],
        "comparison": ">",
        "default": 70,
        "unit": "GeV",
        "fields": ["mass"],
    },
    {
        "id": "quad_max_mass",
        "label": "Four-lepton mass — maximum",
        "description": "Keeps candidates below the selected four-lepton mass.",
        "modes": ["four"],
        "comparison": "<",
        "default": 200,
        "unit": "GeV",
        "fields": ["mass"],
    },
]


class FilterSpec(BaseModel):
    variable: str
    op: Literal["==", "!=", ">", ">=", "<", "<=", "between", "in"]
    value: float | int | str | list[float | int | str] | None = None
    high: float | int | None = None


class Hist1DRequest(BaseModel):
    collection: str
    variable: str
    bins: int = Field(default=60, ge=5, le=300)
    range: list[float] | None = None
    filters: list[FilterSpec] = Field(default_factory=list)


class Hist2DRequest(BaseModel):
    collection: str
    x: str
    y: str
    bins_x: int = Field(default=50, ge=5, le=150)
    bins_y: int = Field(default=50, ge=5, le=150)
    range_x: list[float] | None = None
    range_y: list[float] | None = None
    filters: list[FilterSpec] = Field(default_factory=list)


class RequirementSpec(BaseModel):
    id: str
    value: float | None = None

    @field_validator("id", mode="before")
    @classmethod
    def normalize_requirement_id(cls, value: Any) -> str:
        requirement_id = str(value)
        return LEGACY_REQUIREMENT_IDS.get(requirement_id, requirement_id)


class FormulaRequest(BaseModel):
    mode: Literal["pairs", "four"] = "pairs"
    formula: str = "sqrt(E^2 - px^2 - py^2 - pz^2)"
    pairing_rule: str = "closest_distance"
    channel: Literal["all", "4e", "4mu", "2e2mu"] = "all"
    requirements: list[RequirementSpec] = Field(default_factory=list)
    event_index: int | None = None
    start_after: int | None = None
    limit: int = Field(default=1, ge=1, le=5000)

    @field_validator("pairing_rule", mode="before")
    @classmethod
    def normalize_pairing_rule(cls, value: Any) -> str:
        rule = str(value or "closest_distance")
        rule = LEGACY_PAIRING_RULES.get(rule, rule)
        if rule not in PAIRING_RULES:
            allowed = ", ".join(sorted(PAIRING_RULES))
            raise ValueError(f"Unknown pairing rule {value!r}. Allowed values: {allowed}")
        return rule


class PairingHistRequest(BaseModel):
    strategy: Literal["first", "ossf", "target"] = "first"
    target: float = 91.1876
    bins: int = Field(default=60, ge=10, le=200)
    range: list[float] = Field(default_factory=lambda: [40.0, 140.0])


class CompareHistRequest(BaseModel):
    collection: Literal["Dileptons", "QuadLeptons"]
    variable: str
    bins: int = Field(default=70, ge=10, le=200)
    range: list[float] | None = None
    base_filters: list[FilterSpec] = Field(default_factory=list)
    selection_filters: list[FilterSpec] = Field(default_factory=list)


class FormulaHistRequest(BaseModel):
    collection: Literal["Dileptons", "QuadLeptons"]
    formula: str
    bins: int = Field(default=70, ge=10, le=200)
    range: list[float] | None = None
    filters: list[FilterSpec] = Field(default_factory=list)


class DataStore:
    def __init__(self, root_path: Path):
        if not root_path.exists():
            raise FileNotFoundError(
                f"Could not find ROOT file: {root_path}. Set MASTERCLASS_ROOT or place zz4l_masterclass.root next to the app."
            )
        self.root_path = root_path
        self.metadata = self._load_metadata()
        self.data: dict[str, dict[str, np.ndarray]] = {}
        self.load_warnings: list[str] = []
        self.event_indices: np.ndarray = np.array([], dtype=np.int64)
        self._load()

    def _load_metadata(self) -> dict[str, Any]:
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"description": "No metadata JSON found.", "collections": {}, "variables": {}}

    def _load(self) -> None:
        with uproot.open(self.root_path) as f:
            for collection in COLLECTION_ORDER:
                if collection not in f:
                    continue
                tree = f[collection]
                try:
                    arrays = tree.arrays(library="np")
                    self.data[collection] = {name: arr for name, arr in arrays.items()}
                except (OSError, ValueError, zlib.error) as exc:
                    # A partially transferred ROOT file should still expose every readable
                    # collection/branch so the facilitator receives a useful diagnostic.
                    readable: dict[str, np.ndarray] = {}
                    skipped: list[str] = []
                    for name in tree.keys():
                        try:
                            readable[name] = tree[name].array(library="np")
                        except (OSError, ValueError, zlib.error):
                            skipped.append(name)
                    if readable:
                        self.data[collection] = readable
                    self.load_warnings.append(
                        f"{collection}: bulk read failed ({exc}); skipped unreadable branches: {', '.join(skipped) or 'none'}"
                    )
        self._enrich_quad_selection_fields()
        self._enrich_pairing_fields()
        self._add_student_aliases()
        if "Events" in self.data and "event_index" in self.data["Events"]:
            self.event_indices = np.array(sorted(np.unique(self.data["Events"]["event_index"])), dtype=np.int64)

    def _enrich_quad_selection_fields(self) -> None:
        """Add teaching-selection variables even for files made by an older producer."""
        q = self.data.get("QuadLeptons")
        leptons = self.data.get("Leptons")
        if not q or not leptons:
            return
        required_q = {"event_index", "lep1_index", "lep2_index", "lep3_index", "lep4_index"}
        required_l = {"event_index", "lepton_index", "pt", "eta", "iso"}
        if not required_q.issubset(q) or not required_l.issubset(leptons):
            self.load_warnings.append("Could not derive four-lepton teaching selections: required lepton/quad fields are missing.")
            return

        factor = np.int64(10_000)
        lepton_keys = leptons["event_index"].astype(np.int64) * factor + leptons["lepton_index"].astype(np.int64)
        order = np.argsort(lepton_keys, kind="stable")
        sorted_keys = lepton_keys[order]
        matched: list[np.ndarray] = []
        for name in ("lep1_index", "lep2_index", "lep3_index", "lep4_index"):
            keys = q["event_index"].astype(np.int64) * factor + q[name].astype(np.int64)
            positions = np.searchsorted(sorted_keys, keys)
            if np.any(positions >= len(sorted_keys)) or np.any(sorted_keys[np.minimum(positions, len(sorted_keys) - 1)] != keys):
                self.load_warnings.append("Could not derive four-lepton teaching selections: lepton lookup mismatch.")
                return
            matched.append(order[positions])

        pts = np.stack([leptons["pt"][idx] for idx in matched], axis=1)
        pts = np.sort(pts, axis=1)[:, ::-1]
        etas = np.stack([leptons["eta"][idx] for idx in matched], axis=1)
        isolations = np.stack([leptons["iso"][idx] for idx in matched], axis=1)
        q.setdefault("leading_lepton_pt", pts[:, 0].astype(np.float32))
        q.setdefault("subleading_lepton_pt", pts[:, 1].astype(np.float32))
        q.setdefault("min_lepton_pt", pts[:, -1].astype(np.float32))
        q.setdefault("max_abs_lepton_eta", np.max(np.abs(etas), axis=1).astype(np.float32))
        q.setdefault("max_lepton_iso", np.max(isolations, axis=1).astype(np.float32))
        if "phi" in leptons:
            phis = np.stack([leptons["phi"][idx] for idx in matched], axis=1)
            pair_indices = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
            separations = []
            angles = []
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
            q.setdefault("dr", np.mean(np.stack(separations, axis=1), axis=1).astype(np.float32))
            q.setdefault("opening_angle", np.mean(np.stack(angles, axis=1), axis=1).astype(np.float32))
        q.setdefault("pass_pt_hzz", ((pts[:, 0] > 20.0) & (pts[:, 1] > 10.0)).astype(np.int32))
        q.setdefault("pass_isolation_035", ((np.max(isolations, axis=1) < 0.35) | (np.max(isolations, axis=1) < 0)).astype(np.int32))
        if "z1_mass" in q and "z2_mass" in q:
            q.setdefault("pass_z_windows", ((q["z1_mass"] > 40.0) & (q["z1_mass"] < 120.0) & (q["z2_mass"] > 12.0) & (q["z2_mass"] < 120.0)).astype(np.int32))
        if "n_ossf_pairs" in q:
            q.setdefault("has_ossf_pairing", (q["n_ossf_pairs"] >= 2).astype(np.int32))

    def _enrich_pairing_fields(self) -> None:
        """Derive the simple pair properties used by the student pairing rules."""
        pairings = self.data.get("PairingOptions")
        leptons = self.data.get("Leptons")
        if not pairings or not leptons:
            return
        needed_pairing = {
            "event_index",
            "pair_a_lep1_index",
            "pair_a_lep2_index",
            "pair_b_lep1_index",
            "pair_b_lep2_index",
        }
        needed_lepton = {"event_index", "lepton_index", "abs_pdgid", "charge"}
        if not needed_pairing.issubset(pairings) or not needed_lepton.issubset(leptons):
            self.load_warnings.append("Could not derive pairing charge/type fields: required lepton indices are missing.")
            return

        factor = np.int64(10_000)
        lepton_keys = leptons["event_index"].astype(np.int64) * factor + leptons["lepton_index"].astype(np.int64)
        order = np.argsort(lepton_keys, kind="stable")
        sorted_keys = lepton_keys[order]

        def lookup(index_field: str) -> np.ndarray | None:
            keys = pairings["event_index"].astype(np.int64) * factor + pairings[index_field].astype(np.int64)
            positions = np.searchsorted(sorted_keys, keys)
            if len(sorted_keys) == 0 or np.any(positions >= len(sorted_keys)):
                return None
            if np.any(sorted_keys[positions] != keys):
                return None
            return order[positions]

        for prefix in ("pair_a", "pair_b"):
            first = lookup(f"{prefix}_lep1_index")
            second = lookup(f"{prefix}_lep2_index")
            if first is None or second is None:
                self.load_warnings.append("Could not derive pairing charge/type fields: lepton lookup mismatch.")
                return
            q1 = leptons["charge"][first]
            q2 = leptons["charge"][second]
            p1 = leptons["abs_pdgid"][first]
            p2 = leptons["abs_pdgid"][second]
            pairings.setdefault(f"{prefix}_charge_sum", (q1 + q2).astype(np.int32))
            pairings.setdefault(f"{prefix}_is_sf", (p1 == p2).astype(np.int32))
            pairings.setdefault(f"{prefix}_is_os", (q1 != q2).astype(np.int32))
            pairings.setdefault(f"{prefix}_is_ss", (q1 == q2).astype(np.int32))
            if {"eta", "phi"}.issubset(leptons):
                deta = leptons["eta"][first] - leptons["eta"][second]
                dphi = (leptons["phi"][first] - leptons["phi"][second] + np.pi) % (2 * np.pi) - np.pi
                pairings.setdefault(f"{prefix}_dr", np.sqrt(deta * deta + dphi * dphi).astype(np.float32))

    def _add_student_aliases(self) -> None:
        """Expose concise student-facing names without changing the ROOT schema."""
        for data in self.data.values():
            if "scalar_pt_sum" in data:
                data.setdefault("pt_scalar", data["scalar_pt_sum"])
                data.setdefault("pt_total", data["scalar_pt_sum"])
        pairings = self.data.get("PairingOptions", {})
        for prefix in ("pair_a", "pair_b"):
            legacy = f"{prefix}_scalar_pt_sum"
            if legacy in pairings:
                pairings.setdefault(f"{prefix}_pt_scalar", pairings[legacy])
                pairings.setdefault(f"{prefix}_pt_total", pairings[legacy])

    def collections(self) -> list[str]:
        return list(self.data.keys())

    def variables(self, collection: str) -> list[str]:
        self.require_collection(collection)
        return list(self.data[collection].keys())

    def require_collection(self, collection: str) -> None:
        if collection not in self.data:
            raise HTTPException(status_code=404, detail=f"Unknown collection: {collection}")

    def require_variable(self, collection: str, variable: str) -> None:
        self.require_collection(collection)
        if variable not in self.data[collection]:
            raise HTTPException(status_code=404, detail=f"Unknown variable {variable!r} in collection {collection!r}")

    def mask_from_filters(self, collection: str, filters: list[FilterSpec]) -> np.ndarray:
        self.require_collection(collection)
        n = len(next(iter(self.data[collection].values()))) if self.data[collection] else 0
        mask = np.ones(n, dtype=bool)
        for filt in filters:
            self.require_variable(collection, filt.variable)
            arr = self.data[collection][filt.variable]
            op = filt.op
            val = filt.value
            if op == "==":
                mask &= arr == val
            elif op == "!=":
                mask &= arr != val
            elif op == ">":
                mask &= arr > float(val)  # type: ignore[arg-type]
            elif op == ">=":
                mask &= arr >= float(val)  # type: ignore[arg-type]
            elif op == "<":
                mask &= arr < float(val)  # type: ignore[arg-type]
            elif op == "<=":
                mask &= arr <= float(val)  # type: ignore[arg-type]
            elif op == "between":
                if filt.high is None:
                    raise HTTPException(status_code=400, detail="between filter requires high")
                mask &= (arr >= float(val)) & (arr <= float(filt.high))  # type: ignore[arg-type]
            elif op == "in":
                if not isinstance(val, list):
                    raise HTTPException(status_code=400, detail="in filter requires list value")
                mask &= np.isin(arr, val)
        return mask


store = DataStore(ROOT_PATH)
app = FastAPI(title="ZZ→4l Masterclass Combination Builder")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def disable_browser_cache(request: Any, call_next: Any) -> Any:
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/version")
def version() -> dict[str, str]:
    return {"app_version": APP_VERSION}


@app.get("/api/metadata")
def metadata() -> dict[str, Any]:
    collections: dict[str, Any] = {}
    for collection in store.collections():
        cols = store.variables(collection)
        n_rows = len(next(iter(store.data[collection].values()))) if cols else 0
        collection_meta = store.metadata.get("collections", {}).get(collection, {})
        collections[collection] = {
            **collection_meta,
            "name": collection,
            "n_rows": int(n_rows),
            "variables": cols,
        }
    pair_fields = set(store.data.get("PairingOptions", {}))
    quad_fields = set(store.data.get("QuadLeptons", {}))
    requirement_catalog = []
    for definition in REQUIREMENT_CATALOG:
        fields = set(definition.get("fields", []))
        available = any(
            (mode == "pairs" and fields.issubset(pair_fields)) or
            (mode == "four" and fields.issubset(quad_fields))
            for mode in definition["modes"]
        )
        requirement_catalog.append({k: v for k, v in definition.items() if k != "fields"} | {"available": available})

    return {
        "app_version": APP_VERSION,
        "root_path": str(store.root_path),
        "event_display_url": EVENT_DISPLAY_URL,
        "event_display_url_template": EVENT_DISPLAY_URL_TEMPLATE,
        "load_warnings": store.load_warnings,
        "metadata": store.metadata,
        "collections": collections,
        "requirement_catalog": requirement_catalog,
        "guided_quantities": {
            "Dileptons": ["energy", "momentum", "pt", "pt_scalar", "energy_minus_momentum", "mass", "dr", "opening_angle"],
            "QuadLeptons": ["energy", "momentum", "pt", "pt_scalar", "energy_minus_momentum", "mass", "dr", "opening_angle"],
        },
        "guided_selections": [
            {"id": "charge_zero", "label": "total charge = 0", "variable": "charge_sum", "op": "==", "value": 0, "kind": "physics"},
            {"id": "ossf", "label": "can form two opposite-charge, same-type pairs", "variable": "has_ossf_pairing", "op": "==", "value": 1, "kind": "physics"},
            {"id": "isolation", "label": "largest lepton isolation < 0.35", "variable": "max_lepton_iso", "op": "<", "value": 0.35, "kind": "physics"},
            {"id": "z_windows", "label": "40 < m(Z1) < 120 and 12 < m(Z2) < 120 GeV", "variable": "pass_z_windows", "op": "==", "value": 1, "kind": "physics"},
            {"id": "one_best", "label": "one best four-lepton candidate per event", "variable": "is_best", "op": "==", "value": 1, "kind": "combinatorics"},
            {"id": "pt_thresholds", "label": "leading pT > 20 and second pT > 10 GeV", "variable": "pass_pt_hzz", "op": "==", "value": 1, "kind": "test"},
            {"id": "central", "label": "all four leptons have |eta| < 1.5", "variable": "max_abs_lepton_eta", "op": "<", "value": 1.5, "kind": "test"},
            {"id": "boosted", "label": "four-lepton pT > 50 GeV", "variable": "pt", "op": ">", "value": 50, "kind": "test"},
        ],
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


# -----------------------------
# Ordinary ROOT-column histograms
# -----------------------------

def default_range(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(values, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if lo == hi:
        hi = lo + 1.0
    pad = 0.02 * (hi - lo)
    return float(lo - pad), float(hi + pad)


def peakiness(counts: np.ndarray) -> float:
    nonzero = counts[counts > 0]
    if len(nonzero) == 0:
        return 0.0
    baseline = float(np.median(nonzero))
    if baseline <= 0:
        return 0.0
    return float(np.max(counts) / baseline)


def histogram_payload(values: np.ndarray, bins: int, hist_range: tuple[float, float]) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    counts, edges = np.histogram(values, bins=bins, range=hist_range)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "entries": int(len(values)),
        "bins": centers.tolist(),
        "edges": edges.tolist(),
        "counts": counts.astype(int).tolist(),
        "range": [float(hist_range[0]), float(hist_range[1])],
        "stats": {
            "mean": float(np.mean(values)) if len(values) else None,
            "std": float(np.std(values)) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "peakiness": peakiness(counts),
            "max_bin_center": float(centers[int(np.argmax(counts))]) if len(counts) else None,
        },
    }


@app.post("/api/hist1d")
def hist1d(req: Hist1DRequest) -> dict[str, Any]:
    store.require_variable(req.collection, req.variable)
    arr = store.data[req.collection][req.variable]
    mask = store.mask_from_filters(req.collection, req.filters)
    vals = arr[mask]
    vals = vals[np.isfinite(vals)]
    hist_range = tuple(req.range) if req.range is not None else default_range(vals)
    return {
        "collection": req.collection,
        "variable": req.variable,
        **histogram_payload(vals, req.bins, hist_range),
    }


@app.post("/api/compare_hist")
def compare_hist(req: CompareHistRequest) -> dict[str, Any]:
    store.require_variable(req.collection, req.variable)
    values = store.data[req.collection][req.variable]
    base_mask = store.mask_from_filters(req.collection, req.base_filters)
    selected_mask = base_mask & store.mask_from_filters(req.collection, req.selection_filters)
    base_values = values[base_mask]
    selected_values = values[selected_mask]
    finite_for_range = base_values[np.isfinite(base_values)]
    hist_range = tuple(req.range) if req.range is not None else default_range(finite_for_range)
    return {
        "collection": req.collection,
        "variable": req.variable,
        "before": histogram_payload(base_values, req.bins, hist_range),
        "after": histogram_payload(selected_values, req.bins, hist_range),
        "kept_fraction": float(np.count_nonzero(selected_mask) / max(np.count_nonzero(base_mask), 1)),
        "applied_filters": [filt.model_dump() for filt in req.selection_filters],
    }


@app.post("/api/hist2d")
def hist2d(req: Hist2DRequest) -> dict[str, Any]:
    store.require_variable(req.collection, req.x)
    store.require_variable(req.collection, req.y)
    x = store.data[req.collection][req.x]
    y = store.data[req.collection][req.y]
    mask = store.mask_from_filters(req.collection, req.filters)
    mask &= np.isfinite(x) & np.isfinite(y)
    xv = x[mask]
    yv = y[mask]
    rx = tuple(req.range_x) if req.range_x is not None else default_range(xv)
    ry = tuple(req.range_y) if req.range_y is not None else default_range(yv)
    counts, xedges, yedges = np.histogram2d(xv, yv, bins=[req.bins_x, req.bins_y], range=[rx, ry])
    return {
        "collection": req.collection,
        "x": req.x,
        "y": req.y,
        "entries": int(len(xv)),
        "x_edges": xedges.tolist(),
        "y_edges": yedges.tolist(),
        "z": counts.T.astype(int).tolist(),
        "range_x": [float(rx[0]), float(rx[1])],
        "range_y": [float(ry[0]), float(ry[1])],
    }


@app.post("/api/pairing_hist")
def pairing_hist(req: PairingHistRequest) -> dict[str, Any]:
    """Fast guided-stage histogram without the sandbox's per-event loop."""
    data = store.data.get("PairingOptions", {})
    required = ["event_index", "quad_index", "quad_is_best", "pairing_id", "both_pairs_ossf", "pair_a_mass", "pair_b_mass"]
    missing = [name for name in required if name not in data]
    if missing:
        raise HTTPException(status_code=404, detail=f"PairingOptions is missing: {missing}")

    mask = data["quad_is_best"] == 1
    if req.strategy == "first":
        mask &= data["pairing_id"] == 0
        selected = np.flatnonzero(mask)
    else:
        mask &= data["both_pairs_ossf"] == 1
        eligible = np.flatnonzero(mask)
        if req.strategy == "ossf" or len(eligible) == 0:
            selected = eligible
        else:
            score = np.abs(data["pair_a_mass"][eligible] - req.target) + np.abs(data["pair_b_mass"][eligible] - req.target)
            order = np.argsort(score, kind="stable")
            ranked = eligible[order]
            keys = np.rec.fromarrays(
                [data["event_index"][ranked], data["quad_index"][ranked]],
                names="event_index,quad_index",
            )
            _, first_positions = np.unique(keys, return_index=True)
            selected = ranked[first_positions]

    values = np.concatenate([data["pair_a_mass"][selected], data["pair_b_mass"][selected]]).astype(float)
    values = values[np.isfinite(values)]
    lo, hi = float(req.range[0]), float(req.range[1])
    if not hi > lo:
        raise HTTPException(status_code=422, detail="Histogram range must have max > min")
    counts, edges = np.histogram(values, bins=req.bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "strategy": req.strategy,
        "selected_candidates": int(len(selected)),
        "entries": int(len(values)),
        "bins": centers.tolist(),
        "edges": edges.tolist(),
        "counts": counts.astype(int).tolist(),
        "stats": {
            "peakiness": peakiness(counts),
            "max_bin_center": float(centers[int(np.argmax(counts))]) if len(counts) else None,
        },
    }


# -----------------------------
# Safe formula evaluator
# -----------------------------

ALLOWED_FUNCTIONS = {
    "sqrt": np.sqrt,
    "abs": np.abs,
    "log": np.log,
    "exp": np.exp,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "min": np.minimum,
    "max": np.maximum,
}


def evaluate_formula(expr: str, variables: dict[str, np.ndarray]) -> np.ndarray:
    """Safely evaluate a NumPy expression made from variables, functions, and arithmetic."""
    expr = expr.replace("^", "**")
    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse formula: {exc.msg}") from exc

    def eval_node(node: ast.AST) -> np.ndarray | float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise HTTPException(status_code=400, detail="Formula constants must be numbers")
        if isinstance(node, ast.Name):
            if node.id in variables:
                return variables[node.id]
            raise HTTPException(status_code=400, detail=f"Unknown formula variable: {node.id}")
        if isinstance(node, ast.UnaryOp):
            val = eval_node(node.operand)
            if isinstance(node.op, ast.USub):
                return -val
            if isinstance(node.op, ast.UAdd):
                return val
            raise HTTPException(status_code=400, detail="Unsupported unary operator")
        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                with np.errstate(divide="ignore", invalid="ignore"):
                    return np.asarray(left, dtype=float) / np.asarray(right, dtype=float)
            if isinstance(node.op, ast.Pow):
                return np.power(left, right)
            raise HTTPException(status_code=400, detail="Unsupported operator")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
                raise HTTPException(status_code=400, detail="Unsupported function")
            args = [eval_node(a) for a in node.args]
            if len(args) not in (1, 2):
                raise HTTPException(status_code=400, detail="Functions take one or two arguments")
            with np.errstate(all="ignore"):
                return ALLOWED_FUNCTIONS[node.func.id](*args)
        raise HTTPException(status_code=400, detail="Unsupported formula syntax")

    with np.errstate(all="ignore"):
        result = eval_node(parsed)
    result_arr = np.asarray(result, dtype=float)
    if result_arr.ndim == 0:
        # Broadcast constants to the length of any available variable.
        n = len(next(iter(variables.values()))) if variables else 1
        result_arr = np.full(n, float(result_arr), dtype=float)
    return result_arr


def alias_env(data: dict[str, np.ndarray], prefix: str = "") -> dict[str, np.ndarray]:
    def get(name: str, default: float = np.nan) -> np.ndarray:
        key = f"{prefix}{name}" if prefix else name
        if key in data:
            return data[key].astype(float)
        n = len(next(iter(data.values()))) if data else 0
        return np.full(n, default, dtype=float)

    env = {
        "E": get("energy"),
        "energy": get("energy"),
        "px": get("px"),
        "py": get("py"),
        "pz": get("pz"),
        "p": get("momentum"),
        "momentum": get("momentum"),
        "pt": get("pt"),
        "m": get("mass"),
        "mass": get("mass"),
        "mass2": get("mass_squared"),
        "scalar_pt_sum": get("pt_scalar"),
        "pt_total": get("pt_scalar"),
        "pt_scalar": get("pt_scalar"),
        "charge": get("charge_sum"),
        "charge_sum": get("charge_sum"),
        "dr": get("dr"),
        "dphi": get("dphi"),
        "angle": get("opening_angle"),
        "opening_angle": get("opening_angle"),
    }
    return env


@app.post("/api/formula_hist")
def formula_hist(req: FormulaHistRequest) -> dict[str, Any]:
    data = store.data.get(req.collection, {})
    if not data:
        raise HTTPException(status_code=404, detail=f"Collection {req.collection} is unavailable")
    mask = store.mask_from_filters(req.collection, req.filters)
    values = evaluate_formula(req.formula, alias_env(data))[mask]
    finite = values[np.isfinite(values)]
    hist_range = tuple(req.range) if req.range is not None else default_range(finite)
    return {
        "collection": req.collection,
        "formula": req.formula,
        **histogram_payload(values, req.bins, hist_range),
    }


def rows_for_event(collection: str, event_index: int) -> dict[str, np.ndarray]:
    data = store.data.get(collection, {})
    if not data or "event_index" not in data:
        return {}
    mask = data["event_index"] == event_index
    return {k: v[mask] for k, v in data.items()}


def apply_pair_request_mask(data: dict[str, np.ndarray], req: FormulaRequest) -> np.ndarray:
    n = len(next(iter(data.values()))) if data else 0
    mask = np.ones(n, dtype=bool)
    if "quad_is_best" in data:
        mask &= data["quad_is_best"] == 1
    return mask


def apply_quad_request_mask(data: dict[str, np.ndarray], req: FormulaRequest) -> np.ndarray:
    n = len(next(iter(data.values()))) if data else 0
    mask = np.ones(n, dtype=bool)
    if "is_best" in data:
        mask &= data["is_best"] == 1
    if req.channel != "all" and "channel" in data:
        channel_codes = {"4e": 1, "4mu": 2, "2e2mu": 3}
        mask &= data["channel"] == channel_codes[req.channel]
    return mask


def choose_pairing_index(
    a: np.ndarray,
    b: np.ndarray,
    data: dict[str, np.ndarray],
    req: FormulaRequest,
) -> int | None:
    if len(a) == 0:
        return None
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        return None
    valid = np.flatnonzero(finite)

    def choose_smallest(values: np.ndarray) -> int:
        scores = np.full(len(a), np.inf, dtype=float)
        scores[finite] = values[finite]
        return int(np.argmin(scores))

    if req.pairing_rule in {"closest_distance", "furthest_distance"}:
        da = data.get("pair_a_dr")
        db = data.get("pair_b_dr")
        if da is not None and db is not None:
            distances = np.asarray(da, dtype=float) + np.asarray(db, dtype=float)
            good = finite & np.isfinite(distances)
            if np.any(good):
                finite = good
                return choose_smallest(distances if req.pairing_rule == "closest_distance" else -distances)
    elif req.pairing_rule == "closest_formula":
        return choose_smallest(np.abs(a - b))
    elif req.pairing_rule == "furthest_formula":
        return choose_smallest(-np.abs(a - b))
    elif req.pairing_rule in {
        "same_type_same_charge",
        "same_type_different_charge",
        "different_type_same_charge",
        "different_type_different_charge",
    }:
        same_type = req.pairing_rule.startswith("same_type")
        suffix = "is_ss" if req.pairing_rule.endswith("same_charge") else "is_os"
        needed = ["pair_a_is_sf", "pair_b_is_sf", f"pair_a_{suffix}", f"pair_b_{suffix}"]
        if all(name in data for name in needed):
            matches = (
                (data["pair_a_is_sf"] == int(same_type)) & (data[f"pair_a_{suffix}"] == 1)
            ).astype(int) + (
                (data["pair_b_is_sf"] == int(same_type)) & (data[f"pair_b_{suffix}"] == 1)
            ).astype(int)
            scores = np.full(len(a), np.inf, dtype=float)
            scores[finite] = -matches[finite]
            return int(np.argmin(scores))
    elif req.pairing_rule == "random":
        event = int(data.get("event_index", np.array([0]))[0])
        quad = int(data.get("quad_index", np.array([0]))[0])
        seed = zlib.crc32(f"{event}:{quad}".encode("utf-8"))
        return int(valid[seed % len(valid)])
    return int(valid[0])


def requirement_definition(requirement_id: str) -> dict[str, Any]:
    for definition in REQUIREMENT_CATALOG:
        if definition["id"] == requirement_id:
            return definition
    raise HTTPException(status_code=422, detail=f"Unknown event requirement: {requirement_id}")


def evaluate_event_requirements(
    req: FormulaRequest,
    pair_row: dict[str, Any] | None = None,
    quad_row: dict[str, Any] | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    row = pair_row if req.mode == "pairs" else quad_row
    for selected in req.requirements:
        definition = requirement_definition(selected.id)
        if req.mode not in definition["modes"]:
            raise HTTPException(status_code=422, detail=f"Requirement {selected.id} is not valid in {req.mode} mode")
        threshold = float(definition["default"] if selected.value is None else selected.value)
        values: list[float] = []
        missing = row is None
        if row is not None:
            for field in definition["fields"]:
                raw = row.get(field)
                if raw is None:
                    missing = True
                    break
                try:
                    number = float(raw)
                except (TypeError, ValueError):
                    missing = True
                    break
                if not math.isfinite(number):
                    missing = True
                    break
                values.append(number)

        passed = False
        if not missing:
            if definition["comparison"] == "required":
                expected = definition.get("expected", [1] * len(values))
                passed = len(expected) == len(values) and all(value == wanted for value, wanted in zip(values, expected))
            elif selected.id == "quad_charge_zero":
                passed = abs(values[0]) <= threshold
            elif selected.id == "max_lepton_iso":
                passed = values[0] < threshold or values[0] < 0
            elif definition["comparison"] == ">":
                passed = all(value > threshold for value in values)
            elif definition["comparison"] == "<":
                passed = all(value < threshold for value in values)

        results.append({
            "id": selected.id,
            "label": definition["label"],
            "value": threshold,
            "passed": passed,
            "available": not missing,
        })
    return all(item["passed"] for item in results), results


def _json_value(x: Any) -> Any:
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def table_rows(data: dict[str, np.ndarray], limit: int = 200) -> list[dict[str, Any]]:
    if not data:
        return []
    n = len(next(iter(data.values())))
    rows = []
    for i in range(min(n, limit)):
        rows.append({k: _json_value(v[i]) for k, v in data.items()})
    return rows


def event_display_url(event_row: dict[str, Any], event_index: int) -> str:
    if not EVENT_DISPLAY_URL_TEMPLATE:
        return ""
    return EVENT_DISPLAY_URL_TEMPLATE.format(
        run=event_row.get("run", ""),
        lumi=event_row.get("luminosityBlock", ""),
        event=event_row.get("event", ""),
        event_index=event_index,
    )


def event_row(event_index: int) -> dict[str, Any]:
    events = store.data["Events"]
    mask = events["event_index"] == event_index
    if not np.any(mask):
        raise HTTPException(status_code=404, detail=f"No event_index {event_index}")
    return {k: _json_value(v[mask][0]) for k, v in events.items()}


def event_detail_core(event_index: int, req: FormulaRequest | None = None) -> dict[str, Any]:
    ev = event_row(event_index)
    leptons = table_rows(rows_for_event("Leptons", event_index))
    quads = rows_for_event("QuadLeptons", event_index)
    pairings = rows_for_event("PairingOptions", event_index)

    payload: dict[str, Any] = {
        "event": ev,
        "leptons": leptons,
        "quads": table_rows(quads),
        "pairings": table_rows(pairings),
        "event_display_url": event_display_url(ev, event_index),
    }
    if req is None:
        return payload

    if req.mode == "pairs":
        mask = apply_pair_request_mask(pairings, req)
        pairings = {k: v[mask] for k, v in pairings.items()}
        a_vals = evaluate_formula(req.formula, alias_env(pairings, "pair_a_"))
        b_vals = evaluate_formula(req.formula, alias_env(pairings, "pair_b_"))
        choice = choose_pairing_index(a_vals, b_vals, pairings, req)
        rows = table_rows(pairings)
        for i, row in enumerate(rows):
            row["formula_pair_a"] = _json_value(float(a_vals[i])) if i < len(a_vals) else None
            row["formula_pair_b"] = _json_value(float(b_vals[i])) if i < len(b_vals) else None
            row["selected_by_rule"] = int(choice == i) if choice is not None else 0
        payload["formula_pairings"] = rows
        chosen_row = rows[choice] if choice is not None else None
        passed, requirement_results = evaluate_event_requirements(req, pair_row=chosen_row)
        payload["passes_requirements"] = passed
        payload["requirement_results"] = requirement_results
        payload["selected_values"] = (
            [float(a_vals[choice]), float(b_vals[choice])]
            if choice is not None and passed and np.isfinite(a_vals[choice]) and np.isfinite(b_vals[choice])
            else []
        )
    else:
        mask = apply_quad_request_mask(quads, req)
        quads = {k: v[mask] for k, v in quads.items()}
        vals = evaluate_formula(req.formula, alias_env(quads, "")) if quads else np.array([])
        rows = table_rows(quads)
        for i, row in enumerate(rows):
            row["formula_value"] = _json_value(float(vals[i])) if i < len(vals) else None
            row["selected_by_rule"] = int(i == 0)
        payload["formula_quads"] = rows
        chosen_row = rows[0] if rows else None
        passed, requirement_results = evaluate_event_requirements(req, quad_row=chosen_row)
        payload["passes_requirements"] = passed
        payload["requirement_results"] = requirement_results
        payload["selected_values"] = [float(vals[0])] if len(vals) and np.isfinite(vals[0]) and passed else []
    return payload


@app.get("/api/random_zz_event")
def random_zz_event() -> dict[str, Any]:
    events = store.data.get("Events", {})
    if not events:
        raise HTTPException(status_code=404, detail="Events collection missing")
    if "n_zz_candidate" in events:
        candidate_mask = events["n_zz_candidate"] > 0
    else:
        candidate_mask = np.ones(len(events["event_index"]), dtype=bool)
    event_indices = events["event_index"][candidate_mask]
    if len(event_indices) == 0:
        raise HTTPException(status_code=404, detail="No candidate events found")
    rng = np.random.default_rng()
    event_index = int(rng.choice(event_indices))
    return event_detail_core(event_index)


@app.get("/api/event/{event_index}")
def event_detail(event_index: int) -> dict[str, Any]:
    return event_detail_core(event_index)


@app.post("/api/formula_event")
def formula_event(req: FormulaRequest) -> dict[str, Any]:
    event_index = req.event_index
    if event_index is None:
        return random_formula_event(req)
    return event_detail_core(event_index, req)


def random_formula_event(req: FormulaRequest) -> dict[str, Any]:
    if req.mode == "pairs":
        source = store.data.get("PairingOptions", {})
        mask = apply_pair_request_mask(source, req)
    else:
        source = store.data.get("QuadLeptons", {})
        mask = apply_quad_request_mask(source, req)
    if not source or "event_index" not in source:
        raise HTTPException(status_code=404, detail="No candidate events found")
    candidates = np.unique(source["event_index"][mask])
    if len(candidates) == 0:
        raise HTTPException(status_code=404, detail="No events match the selected mode and channel")
    req.event_index = int(np.random.default_rng().choice(candidates))
    return event_detail_core(req.event_index, req)


@app.post("/api/formula_batch")
def formula_batch(req: FormulaRequest) -> dict[str, Any]:
    if len(store.event_indices) == 0:
        raise HTTPException(status_code=404, detail="No events loaded")

    if req.start_after is None:
        start_pos = 0
    else:
        start_pos = int(np.searchsorted(store.event_indices, req.start_after, side="right"))
        if start_pos >= len(store.event_indices):
            start_pos = 0
    selected_events = store.event_indices[start_pos:start_pos + req.limit]
    if len(selected_events) < req.limit:
        selected_events = np.concatenate([selected_events, store.event_indices[: req.limit - len(selected_events)]])

    values: list[float] = []
    accepted_events = 0
    last_event_index = int(selected_events[-1]) if len(selected_events) else req.start_after

    # Looping over 1000 events is acceptable here because the ROOT file is already loaded into memory.
    for ei in selected_events:
        try:
            detail = event_detail_core(int(ei), req)
        except HTTPException:
            continue
        event_values = [float(v) for v in detail.get("selected_values", []) if v is not None and math.isfinite(float(v))]
        if event_values:
            accepted_events += 1
            values.extend(event_values)

    return {
        "mode": req.mode,
        "formula": req.formula,
        "requested_events": int(len(selected_events)),
        "accepted_events": int(accepted_events),
        "values": values,
        "last_event_index": last_event_index,
    }
