"""
Funzioni di calcolo della GUI XRF (nessuna dipendenza da Streamlit).

Il pacchetto `xrf_model` in ./src non viene modificato: qui si costruisce la
config in memoria, si chiama `run_one_grid` e si fa il post-processing
(scala con la corrente, canali MCA, Poisson, P/B e LLD, lettura/scrittura .mca).
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")      # il pipeline salva un PNG con matplotlib

import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import PchipInterpolator

APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xrf_model.config import load_config                                    # noqa: E402
from xrf_model.pipeline import run_one_grid                                  # noqa: E402
from xrf_model.spectrum import build_spectrum, transmission_be, qe_active_si, sigma_kev  # noqa: E402
from xrf_model.energy_grid import bin_edges, rebin                           # noqa: E402
from xrf_model.output import read_calibration, channel_energies              # noqa: E402
from xrf_model.atmosphere import attenuation_coefficient                     # noqa: E402,F401
from xrf_model.tube import tube_source_term                                  # noqa: E402,F401
from xrf_model.io import load_composition                                    # noqa: E402

CONFIG_PATH = APP_DIR / "configs" / "lab_xrf.yml"
REF_CURRENT_mA = 0.001          # la simulazione gira a 1 µA e poi si scala (modello lineare in I)
MINI_X2_MAX_W = 4.0
WIN_FWHM = 1.2                  # finestra di integrazione picco = 1.2 × FWHM
AG_K_EDGE_keV = 25.514          # bordo K dell'Ag: sotto questa tensione non ci sono righe Ag K
AR_ZONE = (2.8, 3.4)
REMOVED_KEYS = ("detector_response", "mca_calibration", "mca_channels")
FILTER_MATERIALS = ["nessuno", "Al", "Ti", "Ni", "Cu", "Zr", "Mo", "Rh", "Pd", "Ag", "Sn"]
# Dead time: dalla scheda FAST SDD 25 mm², Tpeak = 9.6 µs → 18 kcps in uscita al 50 % di dead time.
# Con un modello paralizzabile (OCR = R·e^(−Rτ)) il 50 % si ha a Rτ = ln2, quindi
# τ = ln2 / (2 · 18 kcps) = 19.3 µs ≈ 2.0 × Tpeak.
TAU_OVER_TPEAK = 2.0
TPEAK_DEFAULT_us = 4.8          # STIMA: il default del DP5 non è riportato nella documentazione disponibile


# ────────────────────────────────────────────────────────────────────────────
# Password
# ────────────────────────────────────────────────────────────────────────────
def check_password(entered: str | None, expected: str | None) -> bool:
    """True se non c'è password impostata, oppure se quella inserita coincide (tempo costante)."""
    import hmac
    if not expected:
        return True
    if entered is None:
        return False
    return hmac.compare_digest(entered.encode("utf-8"), str(expected).encode("utf-8"))


# ────────────────────────────────────────────────────────────────────────────
# Config e campioni
# ────────────────────────────────────────────────────────────────────────────
def file_stamp(path) -> str:
    """mtime + dimensione: entra nella chiave della cache, così un file modificato su disco ricalcola."""
    p = Path(path)
    if not p.exists():
        return "assente"
    s = p.stat()
    return f"{s.st_mtime_ns}-{s.st_size}"


def base_config() -> dict:
    return load_config(CONFIG_PATH)


def local_files_status(cfg: dict) -> dict:
    resp = cfg.get("detector_response")
    resp_ok = bool(resp) and Path(resp["file"]).exists()
    cal = cfg.get("mca_calibration")
    cal_ok = bool(cal) and Path(cal).exists() and bool(cfg.get("mca_channels"))
    return dict(response=resp_ok, calibration=cal_ok,
                response_declared=bool(resp), calibration_declared=bool(cal))


def list_samples(cfg: dict) -> list[str]:
    return sorted(p.stem for p in Path(cfg["paths"]["compositions"]).glob("*.yml"))


def composition_path(cfg: dict, sample: str) -> Path:
    return Path(cfg["paths"]["compositions"]) / f"{sample}.yml"


def normalized(comp: dict) -> dict:
    """Frazioni in massa normalizzate a somma 1 (il file può essere in frazioni o in wt%)."""
    tot = float(sum(comp.values()))
    if tot <= 0:
        raise ValueError("composizione con somma nulla")
    return {k: float(v) / tot for k, v in comp.items()}


def geometry_defaults(cfg: dict) -> dict:
    def ang_dist(x, y, z):
        D = math.sqrt(x * x + y * y + z * z)
        return math.degrees(math.atan2(math.hypot(x, y), z)), D
    ts, Ds = ang_dist(cfg["src_x"], cfg["src_y"], cfg["src_z"])
    td, Dd = ang_dist(cfg["det_x"], cfg["det_y"], cfg["det_z"])
    return dict(theta_src=ts, D_src=Ds, theta_det=td, D_det=Dd)


def geometry_is_parametric(cfg: dict, tol: float = 1e-3) -> tuple[bool, str]:
    """La geometria del config è descrivibile con θ/D (piano x–z, tubo a x<0, rivelatore a x>0, puntato sullo spot)?"""
    r_det = np.array([cfg["det_x"], cfg["det_y"], cfg["det_z"]], float)
    n = np.array(cfg["det_normal"], float)
    n = n / np.linalg.norm(n)
    aim = float(n @ (-r_det / np.linalg.norm(r_det)))
    reasons = []
    if abs(cfg["src_y"]) > tol or abs(cfg["det_y"]) > tol:
        reasons.append("tubo o rivelatore fuori dal piano x–z")
    if not (cfg["src_x"] <= 0 <= cfg["det_x"]):
        reasons.append("tubo e rivelatore non su lati opposti in x")
    if aim < 1 - tol:
        reasons.append(f"rivelatore non puntato sullo spot (cos = {aim:.4f})")
    return (not reasons), "; ".join(reasons)


def build_cfg(p: dict, base: dict | None = None, compositions_dir: str | None = None) -> dict:
    """Config in memoria partendo da lab_xrf.yml, con le sole chiavi reali del pipeline."""
    cfg = copy.deepcopy(base if base is not None else base_config())
    status = local_files_status(cfg)
    cfg["samples"] = [p["sample"]]
    if compositions_dir:
        cfg["paths"]["compositions"] = compositions_dir
    cfg["include_secondary"] = bool(p["secondary"])

    tube = cfg["tube"]
    tube["kvp"] = float(p["kv"])
    tube["transmission_thickness_um"] = float(p["target_um"])
    cfg_filters = {str(m): float(t) for m, t in (tube.get("filters") or {}).items()}
    keep = p.get("keep_filters")
    filters = {m: t for m, t in cfg_filters.items() if keep is None or m in keep}
    if p.get("filter_mat", "nessuno") != "nessuno" and float(p.get("filter_mm", 0)) > 0:
        filters[p["filter_mat"]] = filters.get(p["filter_mat"], 0.0) + float(p["filter_mm"])
    tube["filters"] = filters
    tube["current_mA"] = REF_CURRENT_mA

    if p.get("geo_mode", "parametrica") == "parametrica":
        ts, td = math.radians(p["theta_src"]), math.radians(p["theta_det"])
        cfg["src_x"], cfg["src_y"], cfg["src_z"] = -p["D_src"] * math.sin(ts), 0.0, p["D_src"] * math.cos(ts)
        cfg["det_x"], cfg["det_y"], cfg["det_z"] = p["D_det"] * math.sin(td), 0.0, p["D_det"] * math.cos(td)
        cfg["det_normal"] = [-cfg["det_x"], -cfg["det_y"], -cfg["det_z"]]
    cfg["det_area"] = float(p["det_area_mm2"]) / 100.0
    cfg["atmosphere"] = p["atmosphere"]
    cfg["src_vacuum_path_cm"] = float(p["src_vac"])
    cfg["det_vacuum_path_cm"] = float(p["det_vac"])

    cfg["grid_x_min"], cfg["grid_x_max"] = -p["grid_hx"], p["grid_hx"]
    cfg["grid_y_min"], cfg["grid_y_max"] = -p["grid_hy"], p["grid_hy"]
    cfg["grid_n_x"] = cfg["grid_n_y"] = int(p["grid_n"])

    cfg["be_window_mm"] = float(p["be_um"]) / 1000.0
    cfg["si_active_layer_um"] = float(p["si_um"])
    cfg["detector_fano"] = float(p["fano"])
    cfg["detector_w_eV"] = float(p["w_ev"])
    cfg["detector_noise_eV"] = float(p["noise_ev"])

    if not (p.get("use_response") and status["response"]):
        cfg.pop("detector_response", None)
    # i canali MCA li gestisce la GUI (broadening sui canali + Poisson)
    cfg.pop("mca_calibration", None)
    cfg.pop("mca_channels", None)
    cfg["mca_live_time_s"] = 1.0
    return cfg


def export_config(p: dict, base: dict, uA: float, live_time: float) -> tuple[str, dict]:
    """YAML della config usata + eventuali file accessori {nome: testo} (composizione caricata)."""
    extra = {}
    comp_dir = None
    if p.get("custom_comp"):
        comp_dir = "compositions_gui"
        extra[f"{comp_dir}/{p['sample']}.yml"] = yaml.safe_dump(
            {"composition": p["custom_comp"], "notes": "composizione caricata nella GUI"},
            sort_keys=False, allow_unicode=True)
    cfg = build_cfg(p, base)
    removed = [k for k in REMOVED_KEYS if k in base and k not in cfg]
    cfg["tube"]["current_mA"] = float(uA) / 1000.0
    cfg["mca_live_time_s"] = float(live_time)
    root = cfg.pop("_repo_root", str(APP_DIR))
    for k, v in list(cfg["paths"].items()):
        try:
            cfg["paths"][k] = str(Path(v).relative_to(root))
        except ValueError:
            pass
    cfg["paths"]["output_dir"] = "results/lab_xrf_gui"
    if comp_dir:
        cfg["paths"]["compositions"] = comp_dir
    if "detector_response" in cfg:
        try:
            cfg["detector_response"]["file"] = str(Path(cfg["detector_response"]["file"]).relative_to(root))
        except ValueError:
            pass
    head = [f"# Config esportata dalla GUI XRF il {datetime.now():%Y-%m-%d %H:%M}",
            "# Base: configs/lab_xrf.yml — tube.current_mA e mca_live_time_s = valori della GUI.",
            "# Rumore di Poisson e canali MCA sono calcolati dalla GUI, non dal pipeline."]
    if removed:
        head.append("# Chiavi di lab_xrf.yml rimosse in questa config: " + ", ".join(removed)
                    + " (file non disponibili o canali gestiti dalla GUI).")
    if comp_dir:
        head.append(f"# Composizione caricata: {comp_dir}/{p['sample']}.yml (inclusa nello zip).")
    return "\n".join(head) + "\n" + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), extra


# ────────────────────────────────────────────────────────────────────────────
# Simulazione
# ────────────────────────────────────────────────────────────────────────────
PHYS_KEYS = ["sample", "custom_comp", "secondary", "kv", "target_um", "keep_filters", "filter_mat",
             "filter_mm", "geo_mode", "theta_src", "D_src", "theta_det", "D_det", "det_area_mm2",
             "atmosphere", "src_vac", "det_vac", "grid_hx", "grid_hy", "grid_n", "be_um", "si_um", "fano",
             "w_ev", "noise_ev", "use_response"]


def phys_key(p: dict, base: dict) -> str:
    """Chiave della cache: parametri fisici + impronta di lab_xrf.yml e del file di composizione."""
    d = {k: p.get(k) for k in PHYS_KEYS}
    d["_cfg_stamp"] = file_stamp(CONFIG_PATH)
    d["_comp_stamp"] = ("inline" if p.get("custom_comp")
                        else file_stamp(composition_path(base, p["sample"])))
    return json.dumps(d, sort_keys=True)


def simulate_core(p: dict, base: dict | None = None) -> dict:
    """run_one_grid a 1 µA. Ritorna array e tabelle (tutto lineare nella corrente)."""
    base = base if base is not None else base_config()
    tmp = Path(tempfile.mkdtemp(prefix="xrfgui_"))
    comp_dir = None
    if p.get("custom_comp"):
        comp_dir = tmp / "compositions"
        comp_dir.mkdir()
        (comp_dir / f"{p['sample']}.yml").write_text(
            yaml.safe_dump({"composition": p["custom_comp"]}), encoding="utf-8")
    cfg = build_cfg(p, base, str(comp_dir) if comp_dir else None)
    cfg["paths"]["output_dir"] = str(tmp / "results")
    comp_raw = load_composition(Path(cfg["paths"]["compositions"]) / f"{p['sample']}.yml")
    log = io.StringIO()
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(log):
            res = run_one_grid(None, cfg)[p["sample"]]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    dt = time.time() - t0

    E = np.asarray(res["energy"], float)
    df = res["fluorescence"].copy()
    comp_n = normalized(comp_raw)
    df["Concentration"] = df["Element"].map(comp_n).fillna(0.0)       # frazione normalizzata
    kw = dict(be_mm=cfg["be_window_mm"], si_active_um=cfg["si_active_layer_um"],
              fano=cfg["detector_fano"], noise_ev=cfg["detector_noise_eV"], w_ev=cfg["detector_w_eV"])
    empty = df.iloc[0:0]
    zeros = np.zeros_like(E)
    comp_fluo = build_spectrum(E, zeros, df, **kw)
    comp_coh = build_spectrum(E, np.asarray(res["I_coherent"]), empty, **kw)
    comp_inc = build_spectrum(E, zeros, empty, I_incoherent=np.asarray(res["I_incoherent"]), **kw)
    El = df["E_line_keV"].to_numpy(float)
    df["T_Be_QE"] = transmission_be(El, kw["be_mm"]) * qe_active_si(El, kw["si_active_um"])
    df["cps_det"] = df["Intensity_ph_s"] * df["T_Be_QE"]
    return dict(E=E, sE=np.asarray(res["spectrum_energy"], float), spec=np.asarray(res["spectrum"], float),
                comp_fluo=comp_fluo, comp_coh=comp_coh, comp_inc=comp_inc, fluo=df,
                I_coh=np.asarray(res["I_coherent"]), I_inc=np.asarray(res["I_incoherent"]),
                used_response="detector_response" in cfg, n_valid=int(res["n_valid_cells"]),
                dt=dt, log=log.getvalue(), kw=kw, comp=comp_n, cfg=cfg)


def scaled(sim: dict, uA: float) -> dict:
    s = dict(sim)
    for k in ("spec", "comp_fluo", "comp_coh", "comp_inc", "I_coh", "I_inc"):
        s[k] = sim[k] * uA
    df = sim["fluo"].copy()
    df["Intensity_ph_s"] *= uA
    df["cps_det"] *= uA
    s["fluo"] = df
    return s


# ────────────────────────────────────────────────────────────────────────────
# Canali MCA, broadening, Poisson
# ────────────────────────────────────────────────────────────────────────────
def norm_cdf(x):
    x = np.asarray(x, float)
    from scipy.special import erf
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def edges_from_calib(calib, n):
    half = np.arange(n + 1, dtype=float) - 0.5
    return channel_energies(calib, n, half)


def smooth_rebin(centres, density, new_edges):
    """Conteggi nei nuovi canali da una densità su centri: interpolazione monotona (PCHIP)
    della cumulata, che passa esattamente per i bordi originali → totale conservato, niente scalini."""
    edges = bin_edges(centres)
    cum = np.concatenate([[0.0], np.cumsum(np.clip(density, 0, None) * np.diff(edges))])
    f = PchipInterpolator(edges, cum, extrapolate=False)
    c = f(np.clip(new_edges, edges[0], edges[-1]))
    return np.clip(np.diff(c), 0, None)


def lines_on_channels(df, edges, kw):
    """Righe gaussiane integrate analiticamente su ogni canale (rate per canale, ph/s)."""
    out = np.zeros(len(edges) - 1)
    d = df[(df["cps_det"] > 0) & np.isfinite(df["E_line_keV"])]
    if d.empty:
        return out
    E0 = d["E_line_keV"].to_numpy(float)
    sig = np.asarray(sigma_kev(E0, kw["fano"], kw["noise_ev"], kw["w_ev"]), float)
    I = d["cps_det"].to_numpy(float)
    for e, s, i in zip(E0, sig, I):
        lo, hi = np.searchsorted(edges, [e - 8 * s, e + 8 * s])
        lo, hi = max(lo - 1, 0), min(hi + 1, len(edges))
        if hi - lo < 2:
            continue
        cdf = norm_cdf((edges[lo:hi] - e) / s)
        out[lo:hi - 1] += i * np.diff(cdf)
    return out


def rate_on_channels(sim, edges):
    """Rate per canale [ph/s]. Via gaussiana: continuo PCHIP + righe analitiche sui canali.
    Con la matrice di risposta: PCHIP dello spettro del pipeline."""
    if sim["used_response"]:
        return smooth_rebin(sim["sE"], sim["spec"], edges)
    cont = smooth_rebin(sim["E"], sim["comp_coh"] + sim["comp_inc"], edges)
    return cont + lines_on_channels(sim["fluo"], edges, sim["kw"])


def component_on_channels(sim, which, edges):
    if which == "comp_fluo" and not sim["used_response"]:
        return lines_on_channels(sim["fluo"], edges, sim["kw"])
    return smooth_rebin(sim["E"], sim[which], edges)


def poisson_counts(expected, seed):
    return np.random.default_rng(int(seed)).poisson(np.clip(expected, 0, None)).astype(float)


def residuals(meas_counts, sim_expected, k_meas):
    """Residui (mis − λ)/√λ con λ = attesa simulata riportata sulla scala del misurato.
    k_meas è il fattore che porta il misurato sulla scala del simulato (live time o picco):
    λ = sim_expected / k_meas, così la varianza è quella dei conteggi misurati veri."""
    lam = np.asarray(sim_expected, float) / float(k_meas)
    return (np.asarray(meas_counts, float) - lam) / np.sqrt(np.maximum(lam, 1.0))


def dead_time_fraction(rate_cps, tau_us):
    """Modello paralizzabile: frazione di tempo morto 1 − e^(−R·τ)."""
    return 1.0 - math.exp(-rate_cps * tau_us * 1e-6)


# ────────────────────────────────────────────────────────────────────────────
# .mca: lettura e scrittura
# ────────────────────────────────────────────────────────────────────────────
B_RANGE_keV = (1e-4, 0.2)       # 0.1–200 eV/canale


class McaError(ValueError):
    pass


def _fit_calibration(rows, label):
    pts = np.array(rows, float)
    ch, en = pts[:, 0], pts[:, 1]
    lab = (label or "").lower()
    if "kev" in lab:
        unit = "keV"
    elif re.search(r"\bev\b", lab):
        unit = "eV"
    else:
        unit = "eV" if np.nanmax(np.abs(en)) > 100.0 else "keV"
    en_kev = en * (1e-3 if unit == "eV" else 1.0)
    deg = 2 if len(pts) >= 3 else 1
    coef = np.polyfit(ch, en_kev, deg)
    if deg == 1:
        B, A = coef
        Cq = 0.0
    else:
        Cq, B, A = coef
    return (float(A), float(B), float(Cq)), unit, deg


def parse_mca(raw: bytes) -> dict:
    """Legge un .mca PMCA/Amptek: dati, header, calibrazione, DP5 CONFIGURATION e DPP STATUS."""
    text = raw.decode("latin-1")
    sections, cur = {}, None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("<<") and s.endswith(">>"):
            cur = s.strip("<>").strip().upper()
            if cur.endswith(" END"):          # es. <<DP5 CONFIGURATION END>>
                cur = None
                continue
            sections.setdefault(cur, [])
            continue
        if cur is not None and s:
            sections[cur].append(s)
    if "DATA" not in sections:
        raise McaError("Il file non contiene la sezione <<DATA>>: non sembra un .mca PMCA/Amptek.")
    data = []
    for s in sections["DATA"]:
        try:
            data.append(float(s.split()[0]))
        except ValueError:
            raise McaError(f"Valore non numerico nella sezione <<DATA>>: {s[:40]!r}") from None
    if len(data) < 16:
        raise McaError(f"La sezione <<DATA>> ha solo {len(data)} canali.")
    counts = np.asarray(data, float)

    header = {}
    for s in sections.get("PMCA SPECTRUM", []):
        if " - " in s:
            k, v = s.split(" - ", 1)
            header[k.strip().upper()] = v.strip()

    calib, cal_label, cal_unit, cal_deg, cal_note = None, "", None, None, ""
    rows = []
    for s in sections.get("CALIBRATION", []):
        if s.upper().startswith("LABEL"):
            cal_label = s.split("-", 1)[-1].strip()
            continue
        try:
            rows.append([float(x) for x in s.split()])
        except ValueError:
            pass
    if rows:
        if all(len(r) == 2 for r in rows) and len(rows) >= 2:
            calib, cal_unit, cal_deg = _fit_calibration(rows, cal_label)
        elif len(rows) == 1 and len(rows[0]) == 4:          # formato "1 A B C" in eV
            _, A, B, Cq = rows[0]
            calib, cal_unit, cal_deg = (A * 1e-3, B * 1e-3, Cq * 1e-3), "eV", 2 if Cq else 1
        else:
            cal_note = "sezione <<CALIBRATION>> non riconosciuta"
        if calib is not None and not (B_RANGE_keV[0] <= calib[1] <= B_RANGE_keV[1]):
            cal_note = (f"calibrazione scartata: B = {calib[1]*1000:.4g} eV/canale fuori dall'intervallo fisico "
                        f"{B_RANGE_keV[0]*1000:g}–{B_RANGE_keV[1]*1000:g} eV/canale (unità dedotte: {cal_unit})")
            calib = None

    dp5 = {}
    for s in sections.get("DP5 CONFIGURATION", []):
        for m in re.finditer(r"([A-Z0-9]{3,5})=([^;]*);", s):
            dp5[m.group(1)] = m.group(2).strip()
    status = {}
    for s in sections.get("DPP STATUS", []):
        if ":" in s:
            k, v = s.split(":", 1)
            status[k.strip()] = v.strip()

    def num(x):
        if x is None:
            return None
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(x))
        return float(m.group()) if m else None

    live = num(header.get("LIVE_TIME"))
    real = num(header.get("REAL_TIME")) or num(status.get("Real Time"))
    fast = num(status.get("Fast Count"))
    slow = num(status.get("Slow Count"))
    meas_dt = None
    if fast and slow is not None and fast > 0:
        meas_dt = float(np.clip(1.0 - slow / fast, 0.0, 1.0))
    elif live and real and real > 0:
        meas_dt = float(np.clip(1.0 - live / real, 0.0, 1.0))
    return dict(counts=counts, header=header, calib=calib, cal_label=cal_label, cal_unit=cal_unit,
                cal_deg=cal_deg, cal_note=cal_note, live=live if (live and live > 0) else None,
                live_missing=not (live and live > 0), real=real, dp5=dp5, status=status,
                tpeak_us=num(dp5.get("TPEA")), gain=num(dp5.get("GAIN")), fast=fast, slow=slow,
                meas_dt=meas_dt)


def write_mca(counts, calib, live, real, description) -> bytes:
    A, B, Cq = (1000.0 * c for c in calib)
    out = io.StringIO()
    out.write("<<PMCA SPECTRUM>>\nTAG - \n")
    out.write(f"DESCRIPTION - {description}\nGAIN - 1\nTHRESHOLD - 0\nLIVE_MODE - 0\n")
    out.write(f"PRESET_TIME - Real\nLIVE_TIME - {live:.3f}\nREAL_TIME - {real:.3f}\n")
    out.write(f"START_TIME - {datetime.now():%m/%d/%Y %H:%M:%S}\nSERIAL_NUMBER - \n")
    out.write("<<CALIBRATION>>\nLABEL - Energy\\Channel\n")
    out.write(f"1 {A:.6f} {B:.6f} {Cq:.6e}\n<<DATA>>\n")
    for c in counts:
        out.write(f"{int(c)}\n")
    out.write("<<END>>\n")
    return out.getvalue().encode()


# ────────────────────────────────────────────────────────────────────────────
# Righe, P/B, LLD
# ────────────────────────────────────────────────────────────────────────────
GROUPS = {"Kα": ["Kα1", "Kα2"], "Kβ": ["Kβ1", "Kβ2", "Kβ3"],
          "Lα": ["Lα1", "Lα2"], "Lβ": ["Lβ1", "Lβ2", "Lβ3", "Lβ4", "Lβ15", "Lβ17"]}


def grouped_lines(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for el, d in df.groupby("Element", sort=False):
        for g, names in GROUPS.items():
            sub = d[d["Line"].isin(names)]
            if sub.empty or sub["cps_det"].sum() <= 0:
                continue
            wts = sub["cps_det"]
            rows.append(dict(Element=el, Group=g, Concentration=float(sub["Concentration"].iloc[0]),
                             E=float((sub["E_line_keV"] * wts).sum() / wts.sum()),
                             cps=float(wts.sum()), lines=sub))
    return pd.DataFrame(rows)


def _window_counts(sim, fl, group_row, live):
    kw = sim["kw"]
    E0 = group_row["E"]
    sig = float(sigma_kev(E0, kw["fano"], kw["noise_ev"], kw["w_ev"]))
    fwhm = 2.3548 * sig
    lo, hi = E0 - 0.5 * WIN_FWHM * fwhm, E0 + 0.5 * WIN_FWHM * fwhm
    lines = group_row["lines"]
    si = np.asarray(sigma_kev(lines["E_line_keV"].to_numpy(float), kw["fano"], kw["noise_ev"], kw["w_ev"]))
    frac = norm_cdf((hi - lines["E_line_keV"].to_numpy()) / si) - norm_cdf((lo - lines["E_line_keV"].to_numpy()) / si)
    N = float((lines["cps_det"].to_numpy() * frac).sum() * live)
    B = float(rebin(sim["E"], sim["comp_coh"] + sim["comp_inc"], np.array([lo, hi]))[0] * live)
    others = fl.drop(index=lines.index)
    others = others[(others["E_line_keV"] > lo - 1.0) & (others["E_line_keV"] < hi + 1.0) & (others["cps_det"] > 0)]
    if not others.empty:
        so = np.asarray(sigma_kev(others["E_line_keV"].to_numpy(float), kw["fano"], kw["noise_ev"], kw["w_ev"]))
        eo = others["E_line_keV"].to_numpy()
        B += float((others["cps_det"].to_numpy() * (norm_cdf((hi - eo) / so) - norm_cdf((lo - eo) / so))).sum() * live)
    return N, max(B, 0.0), fwhm


def lld_value(N, B, w, definition):
    """LLD in ppm. '3√B': 3√B/N·w (definita solo con B ≥ 10). 'Currie': L_D = 2.71 + 4.65√B."""
    if N <= 0:
        return np.nan
    if definition == "Currie":
        return (2.71 + 4.65 * math.sqrt(B)) / N * w * 1e6
    return 3.0 * math.sqrt(B) / N * w * 1e6 if B >= 10.0 else np.nan


def lld_table(sim, live, emin, emax, definition="3√B"):
    """Per ogni elemento valuta Kα e Lα (se eccitate e dentro la griglia) e sceglie la LLD migliore. STIMA."""
    fl = sim["fluo"]
    g = grouped_lines(fl)
    if g.empty:
        return pd.DataFrame()
    out = []
    for el, d in g.groupby("Element", sort=False):
        cand = {}
        for grp in ("Kα", "Lα"):
            r = d[d["Group"] == grp]
            if r.empty:
                continue
            r = r.iloc[0]
            if not (emin <= r["E"] <= emax) or r["cps"] <= 0:
                continue
            N, B, fwhm = _window_counts(sim, fl, r, live)
            cand[grp] = dict(row=r, N=N, B=B, fwhm=fwhm, lld=lld_value(N, B, r["Concentration"], definition))
        if not cand:
            continue
        finite = {k: v for k, v in cand.items() if np.isfinite(v["lld"])}
        best_key = min(finite, key=lambda k: finite[k]["lld"]) if finite else max(cand, key=lambda k: cand[k]["N"])
        b = cand[best_key]
        w = b["row"]["Concentration"]
        out.append({"Elemento": el, "Riga usata": best_key, "E [keV]": round(b["row"]["E"], 3),
                    "Conc. [ppm]": w * 1e6, "FWHM [eV]": b["fwhm"] * 1e3,
                    "N picco (finestra)": b["N"], "B fondo (finestra)": b["B"],
                    "P/B": b["N"] / b["B"] if b["B"] > 0 else np.nan,
                    "σ statistico [%]": 100 * math.sqrt(b["N"] + b["B"]) / b["N"] if b["N"] > 0 else np.nan,
                    "LLD Kα [ppm]": cand.get("Kα", {}).get("lld", np.nan),
                    "LLD Lα [ppm]": cand.get("Lα", {}).get("lld", np.nan),
                    f"LLD {definition} [ppm] (STIMA)": b["lld"]})
    return pd.DataFrame(out).sort_values("E [keV]")


def scattering_angle_deg(cfg: dict) -> float:
    """Angolo di scattering allo spot centrale, dai vettori della config (qualsiasi geometria)."""
    r_src = np.array([cfg["src_x"], cfg["src_y"], cfg["src_z"]], float)
    r_det = np.array([cfg["det_x"], cfg["det_y"], cfg["det_z"]], float)
    k_in = -r_src / np.linalg.norm(r_src)
    k_out = r_det / np.linalg.norm(r_det)
    return math.degrees(math.acos(float(np.clip(k_in @ k_out, -1, 1))))


def compton_energy(E, psi_deg):
    return E / (1.0 + E / 510.999 * (1.0 - math.cos(math.radians(psi_deg))))
