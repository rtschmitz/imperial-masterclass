#!/usr/bin/env python3
"""Single-page CMS H->ZZ->4l masterclass backend."""

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
from pydantic import BaseModel, Field


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
APP_VERSION = "3.1.0"
ROOT_PATH = Path(os.environ.get("MASTERCLASS_ROOT", APP_DIR.parent / "data" / "zz4l_masterclass.root")).resolve()
METADATA_PATH = Path(os.environ.get("MASTERCLASS_METADATA", ROOT_PATH.with_suffix(".metadata.json"))).resolve()
EVENT_DISPLAY_URL_TEMPLATE = os.environ.get("EVENT_DISPLAY_URL_TEMPLATE", "")

COLLECTIONS = ("Events", "Leptons", "QuadLeptons")
CHANNEL_CODES = {"all": None, "4e": 1, "4mu": 2, "2e2mu": 3}


class RequirementSpec(BaseModel):
    id: str
    value: float | None = None
    high: float | None = None


class FormulaRequest(BaseModel):
    mode: Literal["pairs", "four"] = "pairs"
    formula: str = "sqrt(E^2-px^2-py^2-pz^2)"
    # Kept as a string so a stale v2 browser can receive a useful compatibility
    # mapping instead of failing request validation before the handler runs.
    pairing_rule: str = "closest_distance"
    save_mode: Literal["both", "larger", "smaller", "sum", "difference"] = "both"
    channel: Literal["all", "4e", "4mu", "2e2mu"] = "all"
    requirements: list[RequirementSpec] = Field(default_factory=list)
    event_index: int | None = None
    start_after: int | None = None
    limit: int = Field(default=1000, ge=1, le=5000)


REQUIREMENT_CATALOG = [
    {
        "id": "selected_pairs_ossf",
        "label": "same-type, opposite-charge pairs",
        "kind": "analysis",
        "modes": ["pairs"],
        "input": "fixed",
        "description": "Z candidates are built from opposite-charge electrons or opposite-charge muons.",
    },
    {
        "id": "charge_zero",
        "label": "four-lepton charge sum = 0",
        "kind": "analysis",
        "modes": ["pairs", "four"],
        "input": "fixed",
        "description": "The H and Z bosons are electrically neutral.",
    },
    {
        "id": "leading_pt_min",
        "label": "leading lepton pT above",
        "kind": "analysis",
        "modes": ["pairs", "four"],
        "input": "min",
        "default": 22.0,
        "unit": "GeV",
        "description": "The analysis requires at least one lepton above 20 GeV; 22 GeV is a reasonable starting test.",
    },
    {
        "id": "subleading_pt_min",
        "label": "second lepton pT above",
        "kind": "analysis",
        "modes": ["pairs", "four"],
        "input": "min",
        "default": 12.0,
        "unit": "GeV",
        "description": "The analysis threshold is 10 GeV; the default is intentionally a little tighter.",
    },
    {
        "id": "max_isolation",
        "label": "largest lepton isolation below",
        "kind": "analysis",
        "modes": ["pairs", "four"],
        "input": "max",
        "default": 0.35,
        "unit": "",
        "description": "Prompt leptons tend to have little nearby activity.",
    },
    {
        "id": "max_abs_eta",
        "label": "all leptons |eta| below",
        "kind": "analysis",
        "modes": ["pairs", "four"],
        "input": "max",
        "default": 2.4,
        "unit": "",
        "description": "Keeps leptons inside the well-instrumented detector acceptance.",
    },
    {
        "id": "z1_mass_min",
        "label": "Z1 invariant mass above",
        "kind": "analysis",
        "modes": ["four"],
        "input": "min",
        "default": 40.0,
        "unit": "GeV",
        "description": "CMS requires the Z candidate closest to 91.2 GeV to have mass above 40 GeV.",
    },
    {
        "id": "z2_mass_min",
        "label": "Z2 invariant mass above",
        "kind": "analysis",
        "modes": ["four"],
        "input": "min",
        "default": 12.0,
        "unit": "GeV",
        "description": "The second Z can be off shell, so its lower threshold is much smaller.",
    },
    {
        "id": "z2_mass_max",
        "label": "Z2 invariant mass below",
        "kind": "analysis",
        "modes": ["four"],
        "input": "max",
        "default": 120.0,
        "unit": "GeV",
        "description": "Upper edge of the broad Z-candidate mass window.",
    },
    {
        "id": "min_os_pair_mass",
        "label": "every opposite-charge pair mass above",
        "kind": "analysis",
        "modes": ["four"],
        "input": "min",
        "default": 4.0,
        "unit": "GeV",
        "description": "Suppresses low-mass resonances; this is an H->ZZ->4l analysis requirement.",
    },
    {
        "id": "min_delta_r",
        "label": "every lepton pair delta-R above",
        "kind": "analysis",
        "modes": ["four"],
        "input": "min",
        "default": 0.02,
        "unit": "",
        "description": "Prevents two reconstructed leptons from occupying effectively the same direction.",
    },
    {
        "id": "four_mass_min",
        "label": "four-lepton invariant mass above",
        "kind": "analysis",
        "modes": ["four"],
        "input": "min",
        "default": 70.0,
        "unit": "GeV",
        "description": "The published analysis starts its four-lepton mass region at 70 GeV.",
    },
    {
        "id": "object_energy_min",
        "label": "combined-object energy above",
        "kind": "explore",
        "modes": ["pairs", "four"],
        "input": "min",
        "default_by_mode": {"pairs": 60.0, "four": 120.0},
        "unit": "GeV",
        "description": "An exploratory cut on E. Test whether the score gain is worth the lost events.",
    },
    {
        "id": "object_pt_min",
        "label": "combined-object pT above",
        "kind": "explore",
        "modes": ["pairs", "four"],
        "input": "min",
        "default_by_mode": {"pairs": 10.0, "four": 20.0},
        "unit": "GeV",
        "description": "An exploratory cut on transverse motion; it is not guaranteed to improve a mass peak.",
    },
    {
        "id": "scalar_pt_sum_min",
        "label": "sum of lepton pT above",
        "kind": "explore",
        "modes": ["pairs", "four"],
        "input": "min",
        "default_by_mode": {"pairs": 40.0, "four": 80.0},
        "unit": "GeV",
        "description": "Adds individual pT values without vector cancellation.",
    },
    {
        "id": "pair_delta_r_max",
        "label": "both selected pairs delta-R below",
        "kind": "explore",
        "modes": ["pairs"],
        "input": "max",
        "default": 3.0,
        "unit": "",
        "description": "An angular-distance hypothesis. Try it, but do not assume closer pairs are always better.",
    },
]
REQUIREMENT_IDS = {item["id"] for item in REQUIREMENT_CATALOG}


class DataStore:
    def __init__(self, root_path: Path):
        if not root_path.exists():
            raise FileNotFoundError(
                f"Could not find ROOT file: {root_path}. Set MASTERCLASS_ROOT or put zz4l_masterclass.root in data/."
            )
        self.root_path = root_path
        self.metadata = self._load_metadata()
        self.data: dict[str, dict[str, np.ndarray]] = {}
        self.load_warnings: list[str] = []
        self._load()
        events = self.data.get("Events", {})
        indices = events.get("event_index", np.array([], dtype=np.int64))
        if "n_quad" in events:
            indices = indices[events["n_quad"] > 0]
        self.event_indices = np.asarray(indices, dtype=np.int64)

    def _load_metadata(self) -> dict[str, Any]:
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return {"description": "No metadata JSON found."}

    def _load(self) -> None:
        with uproot.open(self.root_path) as root_file:
            for collection in COLLECTIONS:
                if collection not in root_file:
                    raise FileNotFoundError(f"Required collection {collection} is absent from {self.root_path}")
                tree = root_file[collection]
                try:
                    arrays = tree.arrays(library="np")
                    self.data[collection] = {name: np.asarray(values) for name, values in arrays.items()}
                except (OSError, ValueError, zlib.error) as exc:
                    readable: dict[str, np.ndarray] = {}
                    skipped: list[str] = []
                    for name in tree.keys():
                        try:
                            readable[name] = np.asarray(tree[name].array(library="np"))
                        except (OSError, ValueError, zlib.error):
                            skipped.append(name)
                    self.data[collection] = readable
                    self.load_warnings.append(
                        f"{collection}: bulk read failed ({exc}); skipped: {', '.join(skipped) or 'none'}"
                    )

    def event_rows(self, collection: str, event_index: int) -> dict[str, np.ndarray]:
        data = self.data.get(collection, {})
        indices = data.get("event_index")
        if indices is None:
            return {}
        left = int(np.searchsorted(indices, event_index, side="left"))
        right = int(np.searchsorted(indices, event_index, side="right"))
        return {name: values[left:right] for name, values in data.items()}


store = DataStore(ROOT_PATH)

app = FastAPI(title="CMS Four-Lepton Masterclass")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def cache_policy(request: Any, call_next: Any) -> Any:
    """Prevent an older localhost app.js from being reused with this backend."""
    response = await call_next(request)
    path = request.url.path
    if path in ("/", "/api/metadata", "/static/app.js", "/static/styles.css"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    elif path == "/static/plotly-2.35.2.min.js":
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return response


def json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def rows_as_dicts(data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    if not data:
        return []
    size = len(next(iter(data.values())))
    return [{name: json_value(values[index]) for name, values in data.items()} for index in range(size)]


def event_row(event_index: int) -> dict[str, Any]:
    rows = rows_as_dicts(store.event_rows("Events", event_index))
    if not rows:
        raise HTTPException(status_code=404, detail=f"No event {event_index}")
    return rows[0]


def event_display_url(event: dict[str, Any], event_index: int) -> str:
    if not EVENT_DISPLAY_URL_TEMPLATE:
        return ""
    return EVENT_DISPLAY_URL_TEMPLATE.format(
        run=event.get("run", ""),
        lumi=event.get("luminosityBlock", ""),
        event=event.get("event", ""),
        event_index=event_index,
    )


def channel_label(code: int | None) -> str:
    return {1: "4e", 2: "4mu", 3: "2e2mu"}.get(code, "other")


def choose_quad(quads: list[dict[str, Any]], channel: str) -> dict[str, Any] | None:
    code = CHANNEL_CODES[channel]
    eligible = [
        quad
        for quad in quads
        if int(quad.get("channel", 0)) in (1, 2, 3)
        and (code is None or int(quad.get("channel", 0)) == code)
    ]
    if not eligible:
        return None
    stored_best = [quad for quad in eligible if int(quad.get("is_best", 0)) == 1]
    if stored_best:
        return stored_best[0]
    return min(
        eligible,
        key=lambda quad: float(quad.get("score_looks_like_ZZ"))
        if quad.get("score_looks_like_ZZ") is not None and math.isfinite(float(quad["score_looks_like_ZZ"]))
        else float("inf"),
    )


def wrapped_delta_phi(phi_a: float, phi_b: float) -> float:
    return math.atan2(math.sin(phi_a - phi_b), math.cos(phi_a - phi_b))


def combine(leptons: list[dict[str, Any]]) -> dict[str, Any]:
    energy = sum(float(lepton.get("energy") or 0.0) for lepton in leptons)
    px = sum(float(lepton.get("px") or 0.0) for lepton in leptons)
    py = sum(float(lepton.get("py") or 0.0) for lepton in leptons)
    pz = sum(float(lepton.get("pz") or 0.0) for lepton in leptons)
    pt = math.hypot(px, py)
    momentum = math.sqrt(px * px + py * py + pz * pz)
    mass_squared = energy * energy - momentum * momentum
    result: dict[str, Any] = {
        "energy": energy,
        "px": px,
        "py": py,
        "pz": pz,
        "pt": pt,
        "momentum": momentum,
        "mass_squared": mass_squared,
        "mass": math.sqrt(max(0.0, mass_squared)),
        "scalar_pt_sum": sum(float(lepton.get("pt") or 0.0) for lepton in leptons),
        "charge_sum": sum(int(lepton.get("charge") or 0) for lepton in leptons),
        "lepton_indices": [int(lepton["lepton_index"]) for lepton in leptons],
    }
    if len(leptons) == 2:
        first, second = leptons
        deta = float(first.get("eta") or 0.0) - float(second.get("eta") or 0.0)
        dphi = wrapped_delta_phi(float(first.get("phi") or 0.0), float(second.get("phi") or 0.0))
        p1 = np.array([float(first.get(axis) or 0.0) for axis in ("px", "py", "pz")])
        p2 = np.array([float(second.get(axis) or 0.0) for axis in ("px", "py", "pz")])
        denominator = float(np.linalg.norm(p1) * np.linalg.norm(p2))
        opening_angle = math.acos(float(np.clip(np.dot(p1, p2) / denominator, -1.0, 1.0))) if denominator else 0.0
        result.update(
            {
                "dr": math.hypot(deta, dphi),
                "dphi": dphi,
                "opening_angle": opening_angle,
                "is_sf": int(first.get("abs_pdgid")) == int(second.get("abs_pdgid")),
                "is_os": int(first.get("charge")) * int(second.get("charge")) < 0,
            }
        )
    return result


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


def formula_environment(obj: dict[str, Any]) -> dict[str, np.ndarray]:
    def value(name: str, fallback: float = np.nan) -> np.ndarray:
        raw = obj.get(name, fallback)
        return np.array([float(raw)], dtype=float)

    return {
        "E": value("energy"),
        "energy": value("energy"),
        "px": value("px"),
        "py": value("py"),
        "pz": value("pz"),
        "p": value("momentum"),
        "momentum": value("momentum"),
        "pt": value("pt"),
        "m": value("mass"),
        "mass": value("mass"),
        "mass2": value("mass_squared"),
        "scalar_pt_sum": value("scalar_pt_sum"),
        "charge": value("charge_sum"),
        "charge_sum": value("charge_sum"),
        "dr": value("dr"),
        "dphi": value("dphi"),
        "angle": value("opening_angle"),
        "opening_angle": value("opening_angle"),
    }


def evaluate_formula(expression: str, variables: dict[str, np.ndarray]) -> np.ndarray:
    expression = expression.replace("^", "**")
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse formula: {exc.msg}") from exc

    def evaluate(node: ast.AST) -> np.ndarray | float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise HTTPException(status_code=400, detail=f"Unknown formula variable: {node.id}")
            return variables[node.id]
        if isinstance(node, ast.UnaryOp):
            operand = evaluate(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return operand
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                with np.errstate(all="ignore"):
                    return np.asarray(left, dtype=float) / np.asarray(right, dtype=float)
            if isinstance(node.op, ast.Pow):
                with np.errstate(all="ignore"):
                    return np.power(left, right)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCTIONS:
            arguments = [evaluate(argument) for argument in node.args]
            if len(arguments) not in (1, 2):
                raise HTTPException(status_code=400, detail="Functions take one or two arguments")
            with np.errstate(all="ignore"):
                return ALLOWED_FUNCTIONS[node.func.id](*arguments)
        raise HTTPException(status_code=400, detail="Unsupported formula syntax")

    result = np.asarray(evaluate(parsed), dtype=float)
    return result if result.ndim else np.array([float(result)])


def formula_value(expression: str, obj: dict[str, Any]) -> float:
    values = evaluate_formula(expression, formula_environment(obj))
    return float(values[0]) if len(values) else float("nan")


def build_pairings(leptons: list[dict[str, Any]], expression: str) -> list[dict[str, Any]]:
    layouts = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
    pairings: list[dict[str, Any]] = []
    for pairing_id, (indices_a, indices_b) in enumerate(layouts):
        pair_a = combine([leptons[index] for index in indices_a])
        pair_b = combine([leptons[index] for index in indices_b])
        pair_a["formula_value"] = formula_value(expression, pair_a)
        pair_b["formula_value"] = formula_value(expression, pair_b)
        pairings.append({"pairing_id": pairing_id, "pair_a": pair_a, "pair_b": pair_b})
    return pairings


def choose_pairing(pairings: list[dict[str, Any]], strategy: str, event_index: int) -> int:
    legacy_aliases = {
        "closest_target": "closest_formula",
        "smallest_difference": "closest_formula",
        "largest_difference": "furthest_formula",
        "smallest_sum": "closest_formula",
        "largest_sum": "furthest_formula",
        "smallest_value": "closest_formula",
        "largest_value": "furthest_formula",
        "first": "closest_distance",
    }
    strategy = legacy_aliases.get(strategy, strategy)
    supported = {
        "closest_distance",
        "furthest_distance",
        "closest_formula",
        "furthest_formula",
        "same_type_same_charge",
        "same_type_different_charge",
        "random",
    }
    if strategy not in supported:
        raise HTTPException(status_code=422, detail=f"Unknown pairing strategy: {strategy}")
    if strategy == "random":
        return int(np.random.default_rng(event_index + 1729).integers(0, len(pairings)))

    def distance(pairing: dict[str, Any]) -> float:
        return float(pairing["pair_a"]["dr"] + pairing["pair_b"]["dr"])

    def formula_gap(pairing: dict[str, Any]) -> float:
        first = float(pairing["pair_a"]["formula_value"])
        second = float(pairing["pair_b"]["formula_value"])
        return abs(first - second) if math.isfinite(first) and math.isfinite(second) else float("inf")

    if strategy == "closest_distance":
        return min(range(len(pairings)), key=lambda index: distance(pairings[index]))
    if strategy == "furthest_distance":
        return max(range(len(pairings)), key=lambda index: distance(pairings[index]))
    if strategy == "closest_formula":
        return min(range(len(pairings)), key=lambda index: formula_gap(pairings[index]))
    if strategy == "furthest_formula":
        return max(range(len(pairings)), key=lambda index: formula_gap(pairings[index]))

    want_opposite = strategy == "same_type_different_charge"

    def matching_pairs(pairing: dict[str, Any]) -> tuple[int, float]:
        count = 0
        for name in ("pair_a", "pair_b"):
            pair = pairing[name]
            if pair["is_sf"] and bool(pair["is_os"]) == want_opposite:
                count += 1
        return count, -distance(pairing)

    return max(range(len(pairings)), key=lambda index: matching_pairs(pairings[index]))


def saved_pair_values(pairing: dict[str, Any], save_mode: str) -> list[float]:
    first = float(pairing["pair_a"]["formula_value"])
    second = float(pairing["pair_b"]["formula_value"])
    if save_mode == "both":
        return [first, second]
    if save_mode == "larger":
        return [max(first, second)]
    if save_mode == "smaller":
        return [min(first, second)]
    if save_mode == "sum":
        return [first + second]
    if save_mode == "difference":
        return [abs(first - second)]
    return []


def candidate_leptons(event_index: int, quad: dict[str, Any]) -> list[dict[str, Any]]:
    all_leptons = rows_as_dicts(store.event_rows("Leptons", event_index))
    lookup = {int(lepton["lepton_index"]): lepton for lepton in all_leptons}
    indices = [int(quad[f"lep{position}_index"]) for position in range(1, 5)]
    try:
        return [lookup[index] for index in indices]
    except KeyError as exc:
        raise HTTPException(status_code=500, detail=f"Lepton lookup failed for event {event_index}") from exc


def physical_pair_context(leptons: list[dict[str, Any]], quad: dict[str, Any]) -> dict[str, float]:
    all_pairs = [combine([leptons[first], leptons[second]]) for first in range(4) for second in range(first + 1, 4)]
    min_dr = min(float(pair["dr"]) for pair in all_pairs)
    os_masses = [float(pair["mass"]) for pair in all_pairs if pair["is_os"]]
    z1 = quad.get("z1_mass")
    z2 = quad.get("z2_mass")
    return {
        "min_dr": min_dr,
        "min_os_mass": min(os_masses) if os_masses else float("nan"),
        "z1_mass": float(z1) if z1 is not None else float("nan"),
        "z2_mass": float(z2) if z2 is not None else float("nan"),
    }


def test_requirements(
    requirements: list[RequirementSpec],
    mode: str,
    leptons: list[dict[str, Any]],
    quad: dict[str, Any],
    four_object: dict[str, Any],
    selected_pairing: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    unknown = [requirement.id for requirement in requirements if requirement.id not in REQUIREMENT_IDS]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown event requirement: {', '.join(unknown)}")

    pts = sorted((float(lepton.get("pt") or 0.0) for lepton in leptons), reverse=True)
    isolations = [float(lepton.get("iso")) for lepton in leptons if lepton.get("iso") is not None and float(lepton["iso"]) >= 0]
    max_isolation = max(isolations) if isolations else -1.0
    max_eta = max(abs(float(lepton.get("eta") or 0.0)) for lepton in leptons)
    physical = physical_pair_context(leptons, quad)
    objects = [four_object]
    if mode == "pairs" and selected_pairing is not None:
        objects = [selected_pairing["pair_a"], selected_pairing["pair_b"]]

    failures: list[str] = []
    for requirement in requirements:
        value = float(requirement.value) if requirement.value is not None else None
        passed = True
        if requirement.id == "selected_pairs_ossf":
            passed = bool(
                selected_pairing
                and all(selected_pairing[name]["is_sf"] and selected_pairing[name]["is_os"] for name in ("pair_a", "pair_b"))
            )
        elif requirement.id == "charge_zero":
            passed = int(four_object["charge_sum"]) == 0
        elif requirement.id == "leading_pt_min":
            passed = value is not None and pts[0] > value
        elif requirement.id == "subleading_pt_min":
            passed = value is not None and pts[1] > value
        elif requirement.id == "max_isolation":
            passed = value is not None and (max_isolation < 0 or max_isolation < value)
        elif requirement.id == "max_abs_eta":
            passed = value is not None and max_eta < value
        elif requirement.id == "z1_mass_min":
            passed = value is not None and physical["z1_mass"] > value
        elif requirement.id == "z2_mass_min":
            passed = value is not None and physical["z2_mass"] > value
        elif requirement.id == "z2_mass_max":
            passed = value is not None and physical["z2_mass"] < value
        elif requirement.id == "min_os_pair_mass":
            passed = value is not None and physical["min_os_mass"] > value
        elif requirement.id == "min_delta_r":
            passed = value is not None and physical["min_dr"] > value
        elif requirement.id == "four_mass_min":
            passed = value is not None and float(four_object["mass"]) > value
        elif requirement.id == "object_energy_min":
            passed = value is not None and all(float(obj["energy"]) > value for obj in objects)
        elif requirement.id == "object_pt_min":
            passed = value is not None and all(float(obj["pt"]) > value for obj in objects)
        elif requirement.id == "scalar_pt_sum_min":
            passed = value is not None and all(float(obj["scalar_pt_sum"]) > value for obj in objects)
        elif requirement.id == "pair_delta_r_max":
            passed = value is not None and selected_pairing is not None and all(
                float(selected_pairing[name]["dr"]) < value for name in ("pair_a", "pair_b")
            )
        if not passed:
            failures.append(requirement.id)
    return not failures, failures


def analysis_event(event_index: int, request: FormulaRequest) -> dict[str, Any]:
    event = event_row(event_index)
    quad = choose_quad(rows_as_dicts(store.event_rows("QuadLeptons", event_index)), request.channel)
    if quad is None:
        raise HTTPException(status_code=404, detail="This event has no candidate in the selected channel")

    leptons = candidate_leptons(event_index, quad)
    four_object = combine(leptons)
    four_object["formula_value"] = formula_value(request.formula, four_object)
    four_object["channel"] = channel_label(int(quad.get("channel", 0)))

    pairings: list[dict[str, Any]] = []
    selected_pairing: dict[str, Any] | None = None
    selected_pairing_index: int | None = None
    raw_values: list[float]
    if request.mode == "pairs":
        pairings = build_pairings(leptons, request.formula)
        selected_pairing_index = choose_pairing(pairings, request.pairing_rule, event_index)
        pairings[selected_pairing_index]["selected"] = True
        selected_pairing = pairings[selected_pairing_index]
        raw_values = saved_pair_values(selected_pairing, request.save_mode)
    else:
        raw_values = [float(four_object["formula_value"])]

    passed, failures = test_requirements(
        request.requirements, request.mode, leptons, quad, four_object, selected_pairing
    )
    finite_values = [value for value in raw_values if math.isfinite(value)]
    return {
        "event": event,
        "event_display_url": event_display_url(event, event_index),
        "leptons": leptons,
        "four_object": four_object,
        "pairings": pairings,
        "selected_pairing_id": selected_pairing_index,
        "candidate_values": finite_values,
        "selected_values": finite_values if passed else [],
        "passes_requirements": passed,
        "failed_requirements": failures,
    }


def random_analysis_event(request: FormulaRequest) -> dict[str, Any]:
    if len(store.event_indices) == 0:
        raise HTTPException(status_code=404, detail="No four-lepton candidates are loaded")
    rng = np.random.default_rng()
    sample_size = min(400, len(store.event_indices))
    for event_index in rng.choice(store.event_indices, size=sample_size, replace=False):
        try:
            return analysis_event(int(event_index), request)
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
    raise HTTPException(status_code=404, detail="No event was found for the selected channel")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/metadata")
def metadata() -> dict[str, Any]:
    sample = store.metadata.get("sample", {})
    return {
        "app_version": APP_VERSION,
        "root_file": store.root_path.name,
        "sample_name": sample.get("sample_name", store.metadata.get("input_file", "unknown sample")),
        "description": store.metadata.get("description", ""),
        "events": int(len(store.data["Events"].get("event_index", []))),
        "candidate_events": int(len(store.event_indices)),
        "load_warnings": store.load_warnings,
        "requirements": REQUIREMENT_CATALOG,
        "pairing_strategies": [
            {"id": "closest_distance", "label": "Closest Distance", "description": "Choose the pairing with the smallest total delta-R."},
            {"id": "furthest_distance", "label": "Furthest Distance", "description": "Choose the pairing with the largest total delta-R."},
            {"id": "closest_formula", "label": "Closest Value from Formula", "description": "Choose the two formula values that are most similar."},
            {"id": "furthest_formula", "label": "Furthest Value from Formula", "description": "Choose the two formula values that differ the most."},
            {"id": "same_type_same_charge", "label": "Same Type, Same Charge", "description": "Prefer same-flavour, same-sign pairs."},
            {"id": "same_type_different_charge", "label": "Same Type, Different Charge", "description": "Prefer same-flavour, opposite-sign pairs."},
            {"id": "random", "label": "Random", "description": "Choose one of the three pairings at random for each event."},
        ],
    }


@app.post("/api/formula_event")
def formula_event(request: FormulaRequest) -> dict[str, Any]:
    if request.event_index is None:
        return random_analysis_event(request)
    return analysis_event(request.event_index, request)


@app.post("/api/formula_batch")
def formula_batch(request: FormulaRequest) -> dict[str, Any]:
    if len(store.event_indices) == 0:
        raise HTTPException(status_code=404, detail="No events are loaded")
    start = 0 if request.start_after is None else int(np.searchsorted(store.event_indices, request.start_after, side="right"))
    if start >= len(store.event_indices):
        start = 0
    selected = store.event_indices[start : start + request.limit]
    if len(selected) < request.limit:
        selected = np.concatenate([selected, store.event_indices[: request.limit - len(selected)]])

    values: list[float] = []
    accepted = 0
    failed_requirements = 0
    missing_channel = 0
    for event_index in selected:
        try:
            detail = analysis_event(int(event_index), request)
        except HTTPException as exc:
            if exc.status_code == 404:
                missing_channel += 1
                continue
            raise
        if detail["passes_requirements"]:
            accepted += 1
            values.extend(detail["selected_values"])
        else:
            failed_requirements += 1

    return {
        "requested_events": int(len(selected)),
        "accepted_events": accepted,
        "failed_requirements": failed_requirements,
        "missing_channel": missing_channel,
        "values": values,
        "last_event_index": int(selected[-1]) if len(selected) else request.start_after,
    }
