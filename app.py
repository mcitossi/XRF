"""
XRF sim — interfaccia grafica Streamlit
==================================================================

GUI per il simulatore a parametri fondamentali (pacchetto `xrf_model`).
Non modifica il pacchetto: lo importa da ./src, costruisce la config in
memoria partendo da configs/lab_xrf.yml e chiama `run_one_grid` così com'è.

Cose che fa l'app (non il pacchetto):
  * scala i risultati con la corrente del tubo (il modello è lineare in I)
  * campiona il rumore di Poisson sui canali MCA (seed impostabile)
  * rebinna lo spettro sui canali di uno spettro misurato caricato
  * stima picco/fondo e LLD (3·√B) — etichettate come STIMA

Avvio:  streamlit run app.py   (dalla cartella del repo, accanto a src/ e configs/)
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")     # il pipeline salva un PNG con matplotlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

st.set_page_config(page_title="XRF sim", page_icon="🌙",
                   layout="wide", initial_sidebar_state="expanded")

try:
    from xrf_model.config import load_config
    from xrf_model.pipeline import run_one_grid
    from xrf_model.spectrum import build_spectrum, transmission_be, qe_active_si, sigma_kev
    from xrf_model.energy_grid import bin_edges, rebin
    from xrf_model.output import read_calibration, channel_energies
    from xrf_model.atmosphere import attenuation_coefficient
    from xrf_model.tube import tube_source_term
    from xrf_model.io import load_composition
except Exception as exc:   # pragma: no cover
    st.error("Non trovo il pacchetto `xrf_model`. Metti `app.py` nella cartella del repo "
             f"(accanto a `src/` e `configs/`) e attiva il venv del repo.\n\nErrore: {exc}")
    st.stop()

CONFIG_PATH = APP_DIR / "configs" / "lab_xrf.yml"
REF_CURRENT_mA = 0.001          # la simulazione gira a 1 µA e poi si scala (modello lineare in I)
MINI_X2_MAX_W = 4.0
WIN_FWHM = 1.2                  # finestra di integrazione picco = 1.2 × FWHM
FILTER_MATERIALS = ["nessuno", "Al", "Ti", "Ni", "Cu", "Zr", "Mo", "Rh", "Pd", "Ag", "Sn"]
AR_ZONE = (2.8, 3.4)

C = dict(total="#2DD4BF", fluo="#F472B6", coh="#60A5FA", inc="#FBBF24", meas="#FB923C",
         poisson="rgba(229,233,242,0.55)", grid="rgba(148,163,184,0.12)")
RUN_COLORS = ["#2DD4BF", "#F472B6", "#FBBF24", "#60A5FA", "#A78BFA", "#34D399", "#F87171", "#E5E9F2"]

# ────────────────────────────────────────────────────────────────────────────
# Stile
# ────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
.hero {padding: 1.1rem 1.4rem; border-radius: 16px; margin-bottom: 1rem;
  background: radial-gradient(1200px 300px at 0% 0%, rgba(45,212,191,0.18), transparent 60%),
              radial-gradient(900px 300px at 100% 0%, rgba(244,114,182,0.14), transparent 60%),
              linear-gradient(135deg, #10172b 0%, #0b1122 100%);
  border: 1px solid rgba(148,163,184,0.18);}
.hero h1 {font-size: 1.85rem; margin: 0; letter-spacing: .3px;
  background: linear-gradient(90deg, #E5E9F2 0%, #2DD4BF 55%, #F472B6 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.hero p {margin: .25rem 0 .55rem 0; color: #9AA4BF; font-size: .95rem;}
.badge {display: inline-block; padding: 2px 10px; margin: 2px 6px 0 0; border-radius: 999px;
  font-size: .75rem; color: #CFD6E6; background: rgba(148,163,184,0.12);
  border: 1px solid rgba(148,163,184,0.22);}
.badge.warn {color: #FDE68A; background: rgba(251,191,36,0.10); border-color: rgba(251,191,36,0.35);}
.badge.ok {color: #99F6E4; background: rgba(45,212,191,0.10); border-color: rgba(45,212,191,0.35);}
div[data-testid="stMetric"] {background: linear-gradient(180deg, #131A2E 0%, #0f1526 100%);
  border: 1px solid rgba(148,163,184,0.16); border-radius: 14px; padding: .7rem .9rem;}
div[data-testid="stMetricLabel"] p {color: #9AA4BF; font-size: .8rem;}
.small-note {color: #8C96B0; font-size: .82rem;}
section[data-testid="stSidebar"] {border-right: 1px solid rgba(148,163,184,0.12);}
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# Config di base e campioni
# ────────────────────────────────────────────────────────────────────────────
@st.cache_resource
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


def geometry_defaults(cfg: dict) -> dict:
    def ang_dist(x, y, z):
        D = math.sqrt(x * x + y * y + z * z)
        return math.degrees(math.atan2(math.hypot(x, y), z)), D
    ts, Ds = ang_dist(cfg["src_x"], cfg["src_y"], cfg["src_z"])
    td, Dd = ang_dist(cfg["det_x"], cfg["det_y"], cfg["det_z"])
    return dict(theta_src=ts, D_src=Ds, theta_det=td, D_det=Dd)


# ────────────────────────────────────────────────────────────────────────────
# Costruzione della config in memoria (solo chiavi reali di lab_xrf.yml)
# ────────────────────────────────────────────────────────────────────────────
def build_cfg(p: dict, compositions_dir: str | None = None) -> dict:
    cfg = copy.deepcopy(base_config())
    status = local_files_status(cfg)
    cfg["samples"] = [p["sample"]]
    if compositions_dir:
        cfg["paths"]["compositions"] = compositions_dir
    cfg["include_secondary"] = bool(p["secondary"])

    tube = cfg["tube"]
    tube["kvp"] = float(p["kv"])
    tube["transmission_thickness_um"] = float(p["target_um"])
    filters = {"Be": float(p["tube_be_mm"])}
    if p["filter_mat"] != "nessuno" and p["filter_mm"] > 0:
        filters[p["filter_mat"]] = filters.get(p["filter_mat"], 0.0) + float(p["filter_mm"])
    tube["filters"] = filters
    tube["current_mA"] = REF_CURRENT_mA

    ts, td = math.radians(p["theta_src"]), math.radians(p["theta_det"])
    cfg["src_x"], cfg["src_y"], cfg["src_z"] = -p["D_src"] * math.sin(ts), 0.0, p["D_src"] * math.cos(ts)
    cfg["det_x"], cfg["det_y"], cfg["det_z"] = p["D_det"] * math.sin(td), 0.0, p["D_det"] * math.cos(td)
    cfg["det_normal"] = [-cfg["det_x"], -cfg["det_y"], -cfg["det_z"]]   # rivelatore puntato sullo spot
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

    if not (p["use_response"] and status["response"]):
        cfg.pop("detector_response", None)
    # i canali MCA li gestisce l'app (rebin + Poisson): il .mca del pipeline non serve
    cfg.pop("mca_calibration", None)
    cfg.pop("mca_channels", None)
    cfg["mca_live_time_s"] = 1.0
    return cfg


def cfg_for_download(p: dict) -> str:
    cfg = build_cfg(p)
    cfg["tube"]["current_mA"] = float(p["uA"]) / 1000.0
    cfg["mca_live_time_s"] = float(p.get("live_time", 1.0))
    root = cfg.pop("_repo_root", str(APP_DIR))
    for k, v in list(cfg["paths"].items()):
        try:
            cfg["paths"][k] = str(Path(v).relative_to(root))
        except ValueError:
            pass
    cfg["paths"]["output_dir"] = "results/lab_xrf_gui"
    if "detector_response" in cfg:
        try:
            cfg["detector_response"]["file"] = str(Path(cfg["detector_response"]["file"]).relative_to(root))
        except ValueError:
            pass
    head = (f"# Config esportata dalla GUI XRF il {datetime.now():%Y-%m-%d %H:%M}\n"
            f"# Base: configs/lab_xrf.yml — tube.current_mA e mca_live_time_s = valori della GUI\n"
            f"# Il rumore di Poisson e il rebin sui canali di un misurato li fa la GUI, non il pipeline.\n")
    return head + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)


# ────────────────────────────────────────────────────────────────────────────
# Simulazione (cache sulla combinazione di parametri fisici, corrente esclusa)
# ────────────────────────────────────────────────────────────────────────────
PHYS_KEYS = ["sample", "custom_comp", "secondary", "kv", "target_um", "tube_be_mm", "filter_mat",
             "filter_mm", "theta_src", "D_src", "theta_det", "D_det", "det_area_mm2", "atmosphere",
             "src_vac", "det_vac", "grid_hx", "grid_hy", "grid_n", "be_um", "si_um", "fano",
             "w_ev", "noise_ev", "use_response"]


@st.cache_data(show_spinner=False, max_entries=48)
def simulate(phys_json: str) -> dict:
    """Esegue run_one_grid a 1 µA. Ritorna array e tabelle (tutto lineare in I)."""
    p = json.loads(phys_json)
    tmp = Path(tempfile.mkdtemp(prefix="xrfgui_"))
    comp_dir = None
    if p.get("custom_comp"):
        comp_dir = tmp / "compositions"
        comp_dir.mkdir()
        (comp_dir / f"{p['sample']}.yml").write_text(
            yaml.safe_dump({"composition": p["custom_comp"]}), encoding="utf-8")
    cfg = build_cfg(p, str(comp_dir) if comp_dir else None)
    cfg["paths"]["output_dir"] = str(tmp / "results")
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
    kw = dict(be_mm=cfg["be_window_mm"], si_active_um=cfg["si_active_layer_um"],
              fano=cfg["detector_fano"], noise_ev=cfg["detector_noise_eV"], w_ev=cfg["detector_w_eV"])
    empty = df.iloc[0:0]
    zeros = np.zeros_like(E)
    comp_fluo = build_spectrum(E, zeros, df, **kw)
    comp_coh = build_spectrum(E, np.asarray(res["I_coherent"]), empty, **kw)
    comp_inc = build_spectrum(E, zeros, empty, I_incoherent=np.asarray(res["I_incoherent"]), **kw)

    # rate rivelato per riga: ingresso rivelatore × T_Be × QE_Si (come build_spectrum)
    El = df["E_line_keV"].to_numpy(float)
    df["T_Be_QE"] = transmission_be(El, kw["be_mm"]) * qe_active_si(El, kw["si_active_um"])
    df["cps_det"] = df["Intensity_ph_s"] * df["T_Be_QE"]
    return dict(E=E, sE=np.asarray(res["spectrum_energy"], float), spec=np.asarray(res["spectrum"], float),
                comp_fluo=comp_fluo, comp_coh=comp_coh, comp_inc=comp_inc, fluo=df,
                I_coh=np.asarray(res["I_coherent"]), I_inc=np.asarray(res["I_incoherent"]),
                used_response="detector_response" in cfg, n_valid=int(res["n_valid_cells"]),
                dt=dt, log=log.getvalue(), kw=kw)


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
# MCA: lettura spettri misurati, canali, scrittura
# ────────────────────────────────────────────────────────────────────────────
def parse_mca(raw: bytes) -> dict:
    text = raw.decode("latin-1")
    section, header, data, cal_label, cal_rows = None, {}, [], "", []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("<<") and s.endswith(">>"):
            section = s.strip("<>").upper()
            if section == "END":
                break
            continue
        if not s:
            continue
        if section == "DATA":
            try:
                data.append(float(s.split()[0]))
            except ValueError:
                pass
        elif section == "CALIBRATION":
            if s.upper().startswith("LABEL"):
                cal_label = s.split("-", 1)[-1].strip()
            else:
                try:
                    cal_rows.append([float(x) for x in s.split()])
                except ValueError:
                    pass
        elif section in ("PMCA SPECTRUM", None) and " - " in s:
            k, v = s.split(" - ", 1)
            header[k.strip().upper()] = v.strip()
    counts = np.asarray(data, float)
    calib = None
    if cal_rows:
        lab = cal_label.lower()
        to_kev = 1e-3 if ("ev" in lab and "kev" not in lab) else 1.0
        if all(len(r) == 2 for r in cal_rows) and len(cal_rows) >= 2:      # coppie canale-energia
            ch, en = np.array(cal_rows).T
            B, A = np.polyfit(ch, en, 1)
            calib = (A * to_kev, B * to_kev, 0.0)
        elif len(cal_rows) == 1 and len(cal_rows[0]) == 4:               # formato "1 A B C" in eV (save_mca)
            _, A, B, Cq = cal_rows[0]
            calib = (A * 1e-3, B * 1e-3, Cq * 1e-3)

    def num(key):
        try:
            return float(header[key])
        except (KeyError, ValueError):
            return None
    return dict(counts=counts, header=header, calib=calib, cal_label=cal_label,
                live=num("LIVE_TIME"), real=num("REAL_TIME"))


def edges_from_calib(calib, n):
    half = np.arange(n + 1, dtype=float) - 0.5
    return channel_energies(calib, n, half)


def write_mca(counts, calib, live, real, description) -> bytes:
    A, B, Cq = (1000.0 * c for c in calib)
    out = io.StringIO()
    out.write("<<PMCA SPECTRUM>>\nTAG - \n")
    out.write(f"DESCRIPTION - {description}\nGAIN - 1\nTHRESHOLD - 0\nLIVE_MODE - 0\n")
    out.write(f"PRESET_TIME - Real\nLIVE_TIME - {live:.1f}\nREAL_TIME - {real:.1f}\n")
    out.write(f"START_TIME - {datetime.now():%m/%d/%Y %H:%M:%S}\nSERIAL_NUMBER - \n")
    out.write("<<CALIBRATION>>\nLABEL - Energy\\Channel\n")
    out.write(f"1 {A:.6f} {B:.6f} {Cq:.6e}\n<<DATA>>\n")
    for c in counts:
        out.write(f"{int(c)}\n")
    out.write("<<END>>\n")
    return out.getvalue().encode()


# ────────────────────────────────────────────────────────────────────────────
# Righe, gruppi, P/B, LLD
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


def norm_cdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x) / math.sqrt(2.0)))


def lld_table(sim, live, emin, emax):
    """P/B e LLD per elemento sul gruppo Kα (o Lα se il K non è eccitato). STIMA."""
    kw = sim["kw"]
    fl = sim["fluo"]
    g = grouped_lines(fl)
    if g.empty:
        return pd.DataFrame()
    out = []
    for el, d in g.groupby("Element", sort=False):
        pick = None
        for grp in ("Kα", "Lα"):
            r = d[d["Group"] == grp]
            if not r.empty and emin <= r["E"].iloc[0] <= emax and r["cps"].iloc[0] > 0:
                pick = r.iloc[0]
                break
        if pick is None:
            continue
        E0 = pick["E"]
        sig = float(sigma_kev(E0, kw["fano"], kw["noise_ev"], kw["w_ev"]))
        fwhm = 2.3548 * sig
        lo, hi = E0 - 0.5 * WIN_FWHM * fwhm, E0 + 0.5 * WIN_FWHM * fwhm
        lines = pick["lines"]
        sig_i = sigma_kev(lines["E_line_keV"].to_numpy(float), kw["fano"], kw["noise_ev"], kw["w_ev"])
        frac = norm_cdf((hi - lines["E_line_keV"]) / sig_i) - norm_cdf((lo - lines["E_line_keV"]) / sig_i)
        N = float((lines["cps_det"].to_numpy() * frac).sum() * live)
        # fondo: continuo di scattering (coerente + incoerente) + code di tutte le altre righe
        cont = float(rebin(sim["E"], sim["comp_coh"] + sim["comp_inc"], np.array([lo, hi]))[0] * live)
        others = fl.drop(index=lines.index)
        others = others[(others["E_line_keV"] > lo - 1.0) & (others["E_line_keV"] < hi + 1.0)]
        if not others.empty:
            so = sigma_kev(others["E_line_keV"].to_numpy(float), kw["fano"], kw["noise_ev"], kw["w_ev"])
            fo = norm_cdf((hi - others["E_line_keV"]) / so) - norm_cdf((lo - others["E_line_keV"]) / so)
            cont += float((others["cps_det"].to_numpy() * fo).sum() * live)
        B = max(cont, 1e-12)
        w = pick["Concentration"]
        # con fondo < 10 conteggi 3√B non ha senso (regime di conteggio basso): LLD non definita
        lld = 3.0 * math.sqrt(B) / N * w * 1e6 if (N > 0 and B >= 10.0) else np.nan
        out.append({"Elemento": el, "Riga": pick["Group"], "E [keV]": round(E0, 3),
                    "Conc. [ppm]": w * 1e6, "FWHM [eV]": fwhm * 1e3,
                    "N picco (finestra)": N, "B fondo (finestra)": B, "P/B": N / B,
                    "σ statistico [%]": 100 * math.sqrt(N + B) / N if N > 0 else np.nan,
                    "LLD 3√B [ppm] (STIMA)": lld})
    return pd.DataFrame(out).sort_values("E [keV]")


def scattering_angle_deg(p):
    ts, td = math.radians(p["theta_src"]), math.radians(p["theta_det"])
    k_in = np.array([math.sin(ts), 0.0, -math.cos(ts)])     # dal tubo (x<0) verso lo spot
    k_out = np.array([math.sin(td), 0.0, math.cos(td)])     # dallo spot verso il rivelatore (x>0)
    return math.degrees(math.acos(np.clip(k_in @ k_out, -1, 1)))


def compton_energy(E, psi_deg):
    return E / (1.0 + E / 510.999 * (1.0 - math.cos(math.radians(psi_deg))))


# ────────────────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────────────────
cfg0 = base_config()
status = local_files_status(cfg0)
gdef = geometry_defaults(cfg0)
samples = list_samples(cfg0)

st.markdown(f"""
<div class="hero">
  <h1>🌙 XRF sim</h1>
  <p>Interfaccia per il codice a parametri fondamentali <code>xrf_model</code> (v3) —
  Mini-X2 Ag + X-123 FAST-SDD, INAF Trieste</p>
  <span class="badge">Sherman + secondaria K</span><span class="badge">righe L + Coster–Kronig</span>
  <span class="badge">Rayleigh f0+f′</span><span class="badge">Compton KN×S + Doppler</span>
  <span class="badge">tubo SpekPy</span>
  <span class="badge {'ok' if status['response'] else 'warn'}">matrice Geant4: {'presente' if status['response'] else 'assente → picchi gaussiani'}</span>
  <span class="badge {'ok' if status['calibration'] else 'warn'}">calibrazione MCA: {'presente' if status['calibration'] else 'assente'}</span>
</div>
""", unsafe_allow_html=True)

if (status["response_declared"] and not status["response"]) or (status["calibration_declared"] and not status["calibration"]):
    st.info("ℹ️ `lab_xrf.yml` dichiara la matrice di risposta Geant4 (`detector_response`) e/o la "
            "calibrazione MCA (`mca_calibration`, `mca_channels`), ma i file non sono su questo computer. "
            "Sono stati tolti dalla config in memoria: i picchi sono gaussiani (Fano + rumore) e i canali sono "
            "quelli della griglia di simulazione (30 eV), oppure quelli dello spettro misurato che carichi. "
            "Il file `lab_xrf.yml` non viene toccato.")

# ────────────────────────────────────────────────────────────────────────────
# Sidebar: parametri fisici (form) + acquisizione (immediati)
# ────────────────────────────────────────────────────────────────────────────
sb = st.sidebar
sb.markdown("### ⚙️ Parametri")

with sb.form("phys"):
    st.markdown("**Campione**")
    default_sample = cfg0["samples"][0] if cfg0["samples"][0] in samples else samples[0]
    sample = st.selectbox("Composizione (data/compositions)", samples, index=samples.index(default_sample))
    custom_file = st.file_uploader("…oppure carica un .yml (`composition:` → frazioni in massa)",
                                   type=["yml", "yaml"], key="custom_yml")
    secondary = st.toggle("Fluorescenza secondaria K", value=bool(cfg0.get("include_secondary", True)))

    st.markdown("**Tubo Mini-X2**")
    c1, c2 = st.columns(2)
    kv = c1.slider("Tensione [kV]", 10, 50, int(cfg0["tube"]["kvp"]))
    uA = c2.slider("Corrente [µA]", 5, 200, 50, step=5)
    target_um = st.number_input("Spessore target Ag [µm]", 0.3, 3.0,
                                float(cfg0["tube"]["transmission_thickness_um"]), 0.05,
                                help="lab_xrf.yml: 1.0 µm. Scheda/manuale Amptek Mini-X2: 0.75 ± 0.1 µm. "
                                     "Con 0.75 µm le righe calano del ~21–25 %.")
    tube_be = float(cfg0["tube"].get("filters", {}).get("Be", 0.125))
    c1, c2 = st.columns(2)
    filter_mat = c1.selectbox("Filtro aggiuntivo", FILTER_MATERIALS, index=0,
                              help=f"Oltre alla finestra Be del tubo ({tube_be} mm). Applicato con SpekPy (tube.filters).")
    filter_mm = c2.number_input("Spessore [mm]", 0.0, 2.0, 0.0, 0.005, format="%.3f")

    st.markdown("**Ambiente e griglia**")
    atmosphere = st.radio("Atmosfera", ["air", "vacuum"], horizontal=True,
                          index=0 if cfg0.get("atmosphere", "vacuum") == "air" else 1,
                          format_func=lambda a: "aria" if a == "air" else "vuoto")
    grid_n = st.select_slider("Griglia n × n", options=[1, 2, 3, 4, 5, 6, 8, 10, 15, 20],
                              value=int(cfg0["grid_n_x"]))
    grid_mode = st.radio("Area illuminata", ["config", "collimatore", "manuale"], horizontal=True,
                         format_func={"config": "da config", "collimatore": "da collimatore",
                                      "manuale": "manuale"}.get,
                         help="Il codice non ha un modello di footprint: la griglia È l'area illuminata.")
    beam_D_mm = st.number_input("Diametro fascio sul campione D [mm] (modo collimatore)", 0.5, 20.0, 2.0, 0.1,
                                help="Da lab_xrf.yml: per un foro di diametro d e lunghezza L con uscita a distanza g "
                                     "dallo spot, D sta fra d e d·(1 + g/L). 2 mm = limite inferiore (foro del collimatore).")
    c1, c2 = st.columns(2)
    hx_man = c1.number_input("½ lato x [cm] (manuale)", 0.01, 3.0, float(cfg0["grid_x_max"]), 0.01)
    hy_man = c2.number_input("½ lato y [cm] (manuale)", 0.01, 3.0, float(cfg0["grid_y_max"]), 0.01)

    with st.expander("📐 Geometria (da lab_xrf.yml)"):
        c1, c2 = st.columns(2)
        theta_src = c1.number_input("θ tubo dalla normale [°]", 0.0, 85.0, round(gdef["theta_src"], 2), 0.5)
        D_src = c2.number_input("D tubo–spot [cm]", 0.3, 30.0, round(gdef["D_src"], 3), 0.05)
        theta_det = c1.number_input("θ rivelatore [°]", 0.0, 85.0, round(gdef["theta_det"], 2), 0.5)
        D_det = c2.number_input("D spot–cristallo [cm]", 0.3, 30.0, round(gdef["D_det"], 3), 0.05)
        src_vac = c1.number_input("Tratto in vuoto lato tubo [cm]", 0.0, 10.0,
                                  float(cfg0.get("src_vacuum_path_cm", 0.0)), 0.05)
        det_vac = c2.number_input("Tratto in vuoto lato rivelatore [cm]", 0.0, 10.0,
                                  float(cfg0.get("det_vacuum_path_cm", 0.0)), 0.05)
        st.caption("x = ∓D·sinθ, z = D·cosθ (formule di lab_xrf.yml); rivelatore puntato sullo spot.")

    with st.expander("🔬 Rivelatore X-123 FAST-SDD"):
        c1, c2 = st.columns(2)
        det_area_mm2 = c1.number_input("Area attiva [mm²]", 1.0, 100.0, float(cfg0["det_area"]) * 100, 1.0,
                                       help="25 mm² nominali, 17 mm² collimati (config).")
        be_um = c2.number_input("Finestra Be [µm]", 0.0, 100.0, float(cfg0.get("be_window_mm", 0.0127)) * 1000, 0.1)
        si_um = c1.number_input("Si attivo [µm]", 100.0, 1000.0, 500.0, 10.0,
                                help=f"lab_xrf.yml: {cfg0.get('si_active_layer_um')} µm. Offerta ORTEC e manuale X-123: 500 µm.")
        fano = c2.number_input("Fano F", 0.05, 0.2, float(cfg0.get("detector_fano", 0.118)), 0.001, format="%.4f")
        w_ev = c1.number_input("w [eV]", 3.5, 3.8, float(cfg0.get("detector_w_eV", 3.65)), 0.01)
        noise_ev = c2.number_input("Rumore σ_el [eV rms]", 0.0, 60.0, float(cfg0.get("detector_noise_eV", 12.7)), 0.1)
        use_response = st.toggle("Usa matrice Geant4 se presente", value=True, disabled=not status["response"])

    est = 1.0 + 0.25 * grid_n ** 2
    if grid_n >= 6:
        st.warning(f"Griglia {grid_n}×{grid_n}: circa {est:.0f} s per run.")
    submitted = st.form_submit_button("▶ Simula", type="primary", width="stretch")

sb.markdown("### ⏱️ Acquisizione")
live_time = sb.number_input("Live time [s]", 1.0, 36000.0, 300.0, 10.0)
c1, c2 = sb.columns([1, 1])
poisson_on = c1.toggle("Rumore Poisson", value=True)
seed = c2.number_input("Seed", 0, 10**6, 42, 1)
tau_us = sb.number_input("τ morto efficace [µs] (STIMA)", 0.1, 20.0, 1.0, 0.1,
                         help="Per la stima del dead time: DT = 1 − exp(−R·τ). Prendilo ≈ dal peaking time del DP5.")

# custom composition
custom_comp, sample_name = None, sample
if custom_file is not None:
    try:
        tmpd = Path(tempfile.mkdtemp(prefix="xrfcomp_"))
        tf = tmpd / "custom.yml"
        tf.write_bytes(custom_file.getvalue())
        custom_comp = {k: float(v) for k, v in load_composition(tf).items()}
        sample_name = "custom_" + Path(custom_file.name).stem.replace(" ", "_")
        shutil.rmtree(tmpd, ignore_errors=True)
    except Exception as exc:
        sb.error(f"Composizione non valida: {exc}")

if grid_mode == "config":
    grid_hx, grid_hy = float(cfg0["grid_x_max"]), float(cfg0["grid_y_max"])
elif grid_mode == "collimatore":
    grid_hx = 0.443 * (beam_D_mm / 10.0) / math.cos(math.radians(theta_src))
    grid_hy = 0.443 * (beam_D_mm / 10.0)
else:
    grid_hx, grid_hy = hx_man, hy_man

params = dict(sample=sample_name, custom_comp=custom_comp, secondary=secondary, kv=kv, uA=uA,
              target_um=target_um, tube_be_mm=tube_be, filter_mat=filter_mat, filter_mm=filter_mm,
              theta_src=theta_src, D_src=D_src, theta_det=theta_det, D_det=D_det,
              det_area_mm2=det_area_mm2, atmosphere=atmosphere, src_vac=src_vac, det_vac=det_vac,
              grid_hx=round(grid_hx, 5), grid_hy=round(grid_hy, 5), grid_n=grid_n, be_um=be_um, si_um=si_um,
              fano=fano, w_ev=w_ev, noise_ev=noise_ev, use_response=use_response)

if submitted or "params" not in st.session_state:
    st.session_state["params"] = params
P = dict(st.session_state["params"])
P["live_time"] = live_time
phys_json = json.dumps({k: P[k] for k in PHYS_KEYS}, sort_keys=True)

# anteprima composizione nella sidebar
with sb.expander("🧪 Composizione del campione simulato", expanded=False):
    comp = P["custom_comp"] or load_composition(Path(cfg0["paths"]["compositions"]) / f"{P['sample']}.yml")
    tot = sum(comp.values())
    cdf = pd.DataFrame({"El": list(comp), "wt %": [100 * v / tot for v in comp.values()]})
    cdf["ppm"] = cdf["wt %"] * 1e4
    st.dataframe(cdf.sort_values("wt %", ascending=False), hide_index=True, width="stretch",
                 column_config={"wt %": st.column_config.NumberColumn(format="%.3f"),
                                "ppm": st.column_config.NumberColumn(format="%.0f")})

# ────────────────────────────────────────────────────────────────────────────
# Simulazione
# ────────────────────────────────────────────────────────────────────────────
power_W = P["kv"] * P["uA"] * 1e-3
try:
    with st.spinner(f"Simulazione FP in corso — {P['sample']}, {P['kv']} kV, griglia {P['grid_n']}×{P['grid_n']}…"):
        sim1 = simulate(phys_json)
except Exception as exc:
    st.error("La simulazione è fallita.")
    st.exception(exc)
    st.stop()
sim = scaled(sim1, P["uA"])

# ── spettro misurato (opzionale) ────────────────────────────────────────────
meas = None
with st.expander("📥 Spettro misurato (.mca Amptek/PMCA) — sovrapponi e confronta", expanded=False):
    up = st.file_uploader("File .mca", type=["mca", "MCA", "txt"], key="meas")
    if up is not None:
        try:
            meas = parse_mca(up.getvalue())
            n = len(meas["counts"])
            c1, c2, c3, c4 = st.columns(4)
            if meas["calib"] is None:
                c1.warning("Nessuna <<CALIBRATION>>: inserisci A, B")
            A = c2.number_input("A [keV]", -5.0, 5.0, float(meas["calib"][0]) if meas["calib"] else 0.0, 0.001, format="%.5f")
            B = c3.number_input("B [keV/canale]", 1e-5, 1.0, float(meas["calib"][1]) if meas["calib"] else 0.006, 0.0001, format="%.6f")
            lt_m = c4.number_input("Live time misurato [s]", 0.001, 1e6, float(meas["live"] or 1.0), 1.0)
            meas.update(A=A, B=B, Cq=float(meas["calib"][2]) if meas["calib"] else 0.0, live=lt_m)
            c1, c2 = st.columns(2)
            meas["align"] = c1.toggle("Simula sui canali del misurato", value=True)
            meas["norm"] = c2.radio("Normalizzazione del misurato",
                                    ["nessuna", "live time", "su un picco"], index=1, horizontal=True)
            st.caption(f"{n} canali · LIVE_TIME {meas['live']} s · REAL_TIME {meas['real']} s · "
                       f"calibrazione: {'dal file (' + meas['cal_label'] + ')' if meas['calib'] else 'manuale'} · "
                       f"conteggi totali {meas['counts'].sum():,.0f}")
        except Exception as exc:
            st.error(f"Non riesco a leggere il file: {exc}")
            meas = None

# ── canali MCA ──────────────────────────────────────────────────────────────
if meas is not None and meas.get("align"):
    calib = (meas["A"], meas["B"], meas["Cq"])
    n_ch = len(meas["counts"])
    ch_src = "canali del misurato"
elif status["calibration"]:
    calib = read_calibration(cfg0["mca_calibration"])
    n_ch = int(cfg0["mca_channels"])
    ch_src = "calibrazione MCA della config"
else:
    step = float(np.mean(np.diff(sim["sE"])))
    calib = (float(sim["sE"][0]), step, 0.0)
    n_ch = len(sim["sE"])
    ch_src = f"griglia di simulazione ({step*1000:.0f} eV/canale)"
edges = edges_from_calib(calib, n_ch)
ch_E = 0.5 * (edges[1:] + edges[:-1])
ch_w = np.diff(edges)
expected = rebin(sim["sE"], sim["spec"], edges) * live_time
rng = np.random.default_rng(int(seed))
noisy = rng.poisson(np.clip(expected, 0, None)).astype(float)
counts = noisy if poisson_on else np.round(expected)

total_rate = float(np.trapezoid(sim["spec"], sim["sE"])) if hasattr(np, "trapezoid") else float(np.trapz(sim["spec"], sim["sE"]))
dead = 1.0 - math.exp(-total_rate * tau_us * 1e-6)
fwhm59 = 2.3548 * float(sigma_kev(5.895, P["fano"], P["noise_ev"], P["w_ev"])) * 1000

# ── metriche ────────────────────────────────────────────────────────────────
def compact(x, unit=""):
    for f, suf in ((1e9, " G"), (1e6, " M"), (1e3, " k")):
        if abs(x) >= f:
            return f"{x / f:.3g}{suf}{unit}"
    return f"{x:.0f} {unit}".strip()


m = st.columns(6)
m[0].metric("Conteggi totali", compact(float(counts.sum())),
            help=f"Nel live time di {live_time:g} s, {'con' if poisson_on else 'senza'} Poisson")
m[1].metric("Rate rivelato", compact(total_rate, "cps"), help="Integrale dello spettro rivelato (sopra ~1 keV)")
m[2].metric("Dead time (STIMA)", f"{100*dead:.1f} %", help=f"1−exp(−R·τ), τ = {tau_us} µs; pile-up non modellato")
m[3].metric("Potenza tubo", f"{power_W:.2f} W", delta=f"max {MINI_X2_MAX_W:.0f} W",
            delta_color="inverse" if power_W > MINI_X2_MAX_W else "off")
m[4].metric("FWHM @ 5.9 keV", f"{fwhm59:.0f} eV")
m[5].metric("Tempo di calcolo", f"{sim['dt']:.1f} s", help="Prima volta per kV/filtro: +~7 s per SpekPy. Poi cache.")
if power_W > MINI_X2_MAX_W:
    st.error(f"⚠️ {P['kv']} kV × {P['uA']} µA = {power_W:.2f} W: oltre i 4 W del Mini-X2. "
             f"A {P['kv']} kV la corrente massima è {MINI_X2_MAX_W / P['kv'] * 1000:.0f} µA.")
if P["grid_n"] * P["grid_n"] and sim["n_valid"] < P["grid_n"] ** 2:
    st.warning(f"Solo {sim['n_valid']} celle su {P['grid_n']**2} sono viste da tubo e rivelatore.")

# ────────────────────────────────────────────────────────────────────────────
# Tabs
# ────────────────────────────────────────────────────────────────────────────
tab_spec, tab_lines, tab_cmp, tab_inst, tab_notes = st.tabs(
    ["📈 Spettro", "📋 Righe & LLD", "🔀 Confronto run", "🛰️ Strumento", "📝 Note fisiche"])

psi = scattering_angle_deg(P)
emax_default = float(min(P["kv"], 40.0))

# ── TAB SPETTRO ─────────────────────────────────────────────────────────────
with tab_spec:
    c = st.columns([1.2, 1.2, 2.2, 1.4])
    ymode = c[0].radio("Asse y", ["conteggi/canale", "ph/s/keV"], horizontal=False)
    yscale = c[1].radio("Scala", ["log", "lin"], horizontal=False)
    erange = c[2].slider("Energia [keV]", 0.0, float(P["kv"]), (1.0, emax_default), 0.1)
    show_comp = c[3].multiselect("Componenti", ["fluorescenza", "coerente", "incoerente"],
                                 default=[] if sim["used_response"] else ["coerente", "incoerente"],
                                 disabled=sim["used_response"],
                                 help="Con la matrice Geant4 le componenti non sono separabili.")
    c = st.columns([1, 1, 1, 3])
    show_labels = c[0].toggle("Etichette righe", value=True)
    show_ar = c[1].toggle("Zona Ar", value=True)
    show_expected = c[2].toggle("Valore atteso", value=poisson_on)
    n_labels = c[3].slider("Numero massimo di etichette", 3, 40, 16)

    to_counts = ymode == "conteggi/canale"

    def tr(E, dens):   # densità ph/s/keV sulla griglia E → asse y scelto
        if to_counts:
            return ch_E, rebin(E, dens, edges) * live_time
        return E, dens

    has_meas = meas is not None
    fig = make_subplots(rows=2 if has_meas else 1, cols=1, shared_xaxes=True,
                        row_heights=[0.76, 0.24] if has_meas else [1.0], vertical_spacing=0.04)

    y_counts = counts if to_counts else counts / (live_time * ch_w)
    if poisson_on:
        fig.add_trace(go.Scatter(x=ch_E, y=y_counts, mode="lines", line=dict(color=C["poisson"], width=1, shape="hvh"),
                                 name=f"simulato + Poisson (seed {seed})"), 1, 1)
    if show_expected or not poisson_on:
        x, y = (ch_E, expected) if to_counts else (sim["sE"], sim["spec"])
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=C["total"], width=2),
                                 name="simulato (atteso)" + (" · Geant4" if sim["used_response"] else "")), 1, 1)
    comp_map = {"fluorescenza": ("comp_fluo", C["fluo"]), "coerente": ("comp_coh", C["coh"]),
                "incoerente": ("comp_inc", C["inc"])}
    for name in show_comp:
        key, col = comp_map[name]
        x, y = tr(sim["E"], sim[key])
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=col, width=1.3, dash="dot"),
                                 name=name), 1, 1)

    k_meas = 1.0
    if has_meas:
        m_edges = edges_from_calib((meas["A"], meas["B"], meas["Cq"]), len(meas["counts"]))
        m_E = 0.5 * (m_edges[1:] + m_edges[:-1])
        m_w = np.diff(m_edges)
        if meas["norm"] == "live time":
            k_meas = live_time / meas["live"]
        elif meas["norm"] == "su un picco":
            gl = grouped_lines(sim["fluo"])
            gl = gl[(gl["E"] > max(erange[0], 1.0)) & (gl["E"] < erange[1])].sort_values("cps", ascending=False)
            opts = [f"{r.Element} {r.Group} ({r.E:.2f} keV)" for r in gl.itertuples()]
            if opts:
                choice = st.selectbox("Picco per la normalizzazione", opts)
                E0 = float(choice.split("(")[1].split()[0])
                sig = float(sigma_kev(E0, P["fano"], P["noise_ev"], P["w_ev"]))
                lo, hi = E0 - 1.5 * sig * 2.3548 / 2 * 1.2, E0 + 1.5 * sig * 2.3548 / 2 * 1.2
                sim_w = float(rebin(sim["sE"], sim["spec"], np.array([lo, hi]))[0] * live_time)
                sel = (m_E >= lo) & (m_E <= hi)
                meas_w = float(meas["counts"][sel].sum())
                k_meas = sim_w / meas_w if meas_w > 0 else 1.0
        y_m = meas["counts"] * k_meas if to_counts else meas["counts"] * k_meas / (live_time * m_w)
        fig.add_trace(go.Scatter(x=m_E, y=y_m, mode="lines", line=dict(color=C["meas"], width=1.2, shape="hvh"),
                                 name=f"misurato × {k_meas:.3g}"), 1, 1)
        # residui normalizzati, solo se gli assi coincidono (stessi canali)
        if meas.get("align"):
            sim_ref = expected
            res = (meas["counts"] * k_meas - sim_ref) / np.sqrt(np.maximum(sim_ref, 1.0))
            fig.add_trace(go.Scatter(x=ch_E, y=res, mode="lines", line=dict(color=C["meas"], width=1),
                                     name="(mis − sim)/√sim", showlegend=False), 2, 1)
            fig.update_yaxes(title_text="residui [σ]", row=2, col=1, gridcolor=C["grid"], zeroline=True,
                             zerolinecolor="rgba(229,233,242,0.4)")
        else:
            fig.add_annotation(text="Residui disponibili con «Simula sui canali del misurato»", row=2, col=1,
                               x=0.5, y=0.5, xref="x domain", yref="y2 domain", showarrow=False,
                               font=dict(color="#8C96B0"))

    if show_ar:
        fig.add_vrect(x0=AR_ZONE[0], x1=AR_ZONE[1], fillcolor="rgba(248,113,113,0.10)", line_width=0,
                      annotation_text="Ar K (2.96 keV): non modellato", annotation_position="top left",
                      annotation_font=dict(color="#FCA5A5", size=11), row=1, col=1)

    # etichette righe
    ref_x, ref_y = (ch_E, expected) if to_counts else (sim["sE"], sim["spec"])
    if show_labels:
        gl = grouped_lines(sim["fluo"])
        gl = gl[(gl["E"] >= erange[0]) & (gl["E"] <= erange[1]) & (gl["E"] >= ref_x[0])]
        gl = gl.sort_values("cps", ascending=False).head(n_labels)
        scat = [("Ag Kα · Rayleigh", 22.10), ("Ag Kβ · Rayleigh", 24.94),
                ("Ag Kα · Compton", compton_energy(22.10, psi))]
        items = [(f"{r.Element} {r.Group}", r.E) for r in gl.itertuples()]
        items += [(n, e) for n, e in scat if erange[0] <= e <= erange[1] and e < P["kv"]]
        for label, E0 in items:
            j = np.searchsorted(ref_x, E0)
            lo_j, hi_j = max(j - 3, 0), min(j + 4, len(ref_y))
            yv = float(ref_y[lo_j:hi_j].max()) if hi_j > lo_j else 0.0
            if yv <= 0:
                continue
            ya = math.log10(yv) if yscale == "log" else yv
            col = "#93C5FD" if "Ag" in label else "#E5E9F2"
            fig.add_annotation(x=E0, y=ya, text=label, showarrow=True, arrowhead=0, arrowwidth=1,
                               arrowcolor="rgba(229,233,242,0.35)", ax=0, ay=-26, textangle=-60,
                               font=dict(size=10, color=col), row=1, col=1)

    ytitle = "conteggi / canale" if to_counts else "ph / s / keV"
    fig.update_yaxes(type="log" if yscale == "log" else "linear", title_text=ytitle,
                     gridcolor=C["grid"], row=1, col=1)
    if yscale == "log":
        sel = (ref_x >= erange[0]) & (ref_x <= erange[1])
        ymax = float(np.max(ref_y[sel])) if sel.any() else 1.0
        ymin_floor = 0.5 if to_counts else ymax * 1e-6
        fig.update_yaxes(range=[math.log10(max(ymin_floor, 1e-12)), math.log10(max(ymax, 1e-9) * 6)], row=1, col=1)
    fig.update_xaxes(range=list(erange), gridcolor=C["grid"], row=1, col=1)
    fig.update_xaxes(title_text="Energia [keV]", gridcolor=C["grid"], row=2 if has_meas else 1, col=1)
    fig.update_layout(template="plotly_dark", height=620 if has_meas else 560,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1324",
                      margin=dict(l=60, r=20, t=40, b=50), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                                  bgcolor="rgba(0,0,0,0)"),
                      title=dict(text=f"<b>{P['sample']}</b> · {P['kv']} kV · {P['uA']} µA · "
                                      f"{'aria' if P['atmosphere']=='air' else 'vuoto'} · LT {live_time:g} s · "
                                      f"griglia {P['grid_n']}×{P['grid_n']}", font=dict(size=14), x=0.0, y=0.995))
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "toImageButtonOptions":
                    {"format": "png", "filename": f"xrf_{P['sample']}_{P['kv']}kV", "scale": 2}})
    st.markdown(f"<span class='small-note'>Canali: {ch_src} · {n_ch} canali · angolo di scattering {psi:.0f}° · "
                f"spettro del tubo da {sim['E'][0]:.2f} keV (sotto ~1 keV, Na e Mg Kα, la griglia SpekPy non arriva).</span>",
                unsafe_allow_html=True)

    # ── salvataggio run e download ──────────────────────────────────────────
    st.markdown("#### 💾 Salva ed esporta")
    c = st.columns([2, 1, 1, 1, 1, 1])
    default_label = (f"{P['sample']} · {P['kv']} kV {P['uA']} µA · "
                     f"{'aria' if P['atmosphere']=='air' else 'vuoto'}"
                     + (f" · {P['filter_mat']} {P['filter_mm']} mm" if P['filter_mat'] != 'nessuno' and P['filter_mm'] > 0 else ""))
    run_label = c[0].text_input("Etichetta run", default_label, label_visibility="collapsed")
    if c[1].button("📌 Aggiungi al confronto", width="stretch"):
        runs = st.session_state.setdefault("runs", [])
        gl = grouped_lines(sim["fluo"])
        runs.append(dict(label=run_label, E=sim["sE"], spec=sim["spec"], params=dict(P),
                         lines={f"{r.Element} {r.Group}": r.cps for r in gl.itertuples()},
                         lines_E={f"{r.Element} {r.Group}": r.E for r in gl.itertuples()}))
        st.session_state["runs"] = runs[-8:]
        st.toast(f"Run salvato: {run_label}", icon="📌")

    base = f"xrf_{P['sample']}_{P['kv']}kV_{P['uA']}uA_{P['atmosphere']}"
    desc = f"XRF GUI {P['sample']} {P['kv']}kV {P['uA']}uA {P['atmosphere']} {'Poisson seed ' + str(seed) if poisson_on else 'atteso'}"
    c[2].download_button("⬇ .mca", write_mca(counts, calib, live_time, live_time, desc),
                         file_name=base + ".mca", width="stretch")
    spec_df = pd.DataFrame({"Energy_keV": sim["E"], "fluorescence_ph_s_keV": sim["comp_fluo"],
                            "coherent_ph_s_keV": sim["comp_coh"], "incoherent_ph_s_keV": sim["comp_inc"]})
    spec_df["total_gaussian_ph_s_keV"] = spec_df.iloc[:, 1:].sum(axis=1)
    chan_df = pd.DataFrame({"channel": np.arange(n_ch), "E_keV": ch_E, "expected_counts": expected,
                            "counts": counts})
    csv_buf = io.StringIO()
    csv_buf.write(f"# {desc}; live_time_s={live_time}; canali: {ch_src}\n")
    chan_df.to_csv(csv_buf, index=False)
    csv_buf.write("\n# Densità sulla griglia di simulazione (ph/s/keV, risposta gaussiana)\n")
    spec_df.to_csv(csv_buf, index=False)
    c[3].download_button("⬇ .csv", csv_buf.getvalue().encode(), file_name=base + ".csv", width="stretch")
    xbuf = io.BytesIO()
    lines_out = sim["fluo"].copy()
    lines_out["counts_in_live_time"] = lines_out["cps_det"] * live_time
    with pd.ExcelWriter(xbuf) as xw:
        lines_out.to_excel(xw, sheet_name="Righe", index=False)
        lld_table(sim, live_time, float(sim["sE"][0]), float(P["kv"])).to_excel(xw, sheet_name="LLD (stima)", index=False)
        pd.DataFrame({"parametro": list(P), "valore": [str(v) for v in P.values()]}).to_excel(
            xw, sheet_name="Parametri", index=False)
    c[4].download_button("⬇ .xlsx", xbuf.getvalue(), file_name=base + "_righe.xlsx", width="stretch")
    c[5].download_button("⬇ config", cfg_for_download(P).encode(), file_name=base + ".yml", width="stretch")

# ── TAB RIGHE ───────────────────────────────────────────────────────────────
with tab_lines:
    st.markdown("#### Righe di fluorescenza")
    df = sim["fluo"].copy()
    df = df[df["Intensity_ph_s"] > 0]
    tab = pd.DataFrame({
        "Elemento": df["Element"], "Riga": df["Line"], "E [keV]": df["E_line_keV"],
        "Conc. [ppm]": df["Concentration"] * 1e6,
        "Ingresso riv. [ph/s]": df["Intensity_ph_s"], "T_Be·QE_Si": df["T_Be_QE"],
        "Rivelati [cps]": df["cps_det"], f"Conteggi in {live_time:g} s": df["cps_det"] * live_time})
    min_cps = st.slider("Mostra righe con rate rivelato ≥ [cps]", 0.0, 1.0, 0.001, 0.001, format="%.3f")
    tab = tab[tab["Rivelati [cps]"] >= min_cps].sort_values("Rivelati [cps]", ascending=False)
    st.dataframe(tab, hide_index=True, width="stretch", height=360, column_config={
        "E [keV]": st.column_config.NumberColumn(format="%.4f"),
        "Conc. [ppm]": st.column_config.NumberColumn(format="%.1f"),
        "Ingresso riv. [ph/s]": st.column_config.NumberColumn(format="%.4g"),
        "T_Be·QE_Si": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.3f"),
        "Rivelati [cps]": st.column_config.NumberColumn(format="%.4g"),
        f"Conteggi in {live_time:g} s": st.column_config.NumberColumn(format="%.4g")})

    st.markdown("#### Picco/fondo e limite di rivelabilità  <span class='badge warn'>STIMA</span>",
                unsafe_allow_html=True)
    lt = lld_table(sim, live_time, float(sim["sE"][0]), float(P["kv"]))
    if not lt.empty:
        c1, c2 = st.columns([3, 2])
        c1.dataframe(lt, hide_index=True, width="stretch", height=420, column_config={
            "Conc. [ppm]": st.column_config.NumberColumn(format="%.1f"),
            "FWHM [eV]": st.column_config.NumberColumn(format="%.0f"),
            "N picco (finestra)": st.column_config.NumberColumn(format="%.3g"),
            "B fondo (finestra)": st.column_config.NumberColumn(format="%.3g"),
            "P/B": st.column_config.NumberColumn(format="%.3g"),
            "σ statistico [%]": st.column_config.NumberColumn(format="%.2f"),
            "LLD 3√B [ppm] (STIMA)": st.column_config.NumberColumn(format="%.3g")})
        lf = go.Figure(go.Bar(x=lt["Elemento"] + " " + lt["Riga"], y=lt["LLD 3√B [ppm] (STIMA)"],
                              marker=dict(color=lt["E [keV]"], colorscale="Tealrose", showscale=False),
                              hovertemplate="%{x}: %{y:.3g} ppm<extra></extra>"))
        lf.update_layout(template="plotly_dark", height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1324",
                         yaxis=dict(type="log", title="LLD [ppm]", gridcolor=C["grid"]),
                         margin=dict(l=50, r=10, t=30, b=40), title=dict(text=f"LLD (3√B) in {live_time:g} s", font=dict(size=13)))
        c2.plotly_chart(lf, width="stretch", config={"displaylogo": False})
    st.caption(f"Finestra = {WIN_FWHM} × FWHM centrata sul gruppo Kα (o Lα se il K non è eccitato). "
               "N = conteggi netti del gruppo nella finestra; B = tutto il resto nella finestra (continuo di "
               "scattering + code di altre righe). LLD = 3√B / N × concentrazione, per risposta lineare. "
               "Se B < 10 conteggi la LLD 3√B non è definita (cella vuota): sotto ~2 keV in aria il continuo simulato è quasi nullo "
               "perché manca il fondo reale (shelf, rumore, Ar). Senza Ar, pile-up, escape e shelf (con picchi gaussiani): ordine di grandezza, non un valore di specifica.")

# ── TAB CONFRONTO ───────────────────────────────────────────────────────────
with tab_cmp:
    runs = st.session_state.get("runs", [])
    if not runs:
        st.info("Nessun run salvato. Nel tab «Spettro» premi **📌 Aggiungi al confronto** (per esempio "
                "una volta in aria e una nel vuoto, oppure con due campioni).")
    else:
        c1, c2, c3 = st.columns([1, 1, 3])
        cmp_scale = c1.radio("Scala ", ["log", "lin"], horizontal=True, key="cmp_scale")
        cmp_norm = c2.toggle("Normalizza per µA", value=False)
        if c3.button("🗑️ Svuota il confronto"):
            st.session_state["runs"] = []
            st.rerun()
        fig2 = go.Figure()
        for i, r in enumerate(runs):
            k = 1.0 / r["params"]["uA"] if cmp_norm else 1.0
            fig2.add_trace(go.Scatter(x=r["E"], y=r["spec"] * k, mode="lines", name=r["label"],
                                      line=dict(color=RUN_COLORS[i % len(RUN_COLORS)], width=1.6)))
        if show_ar:
            fig2.add_vrect(x0=AR_ZONE[0], x1=AR_ZONE[1], fillcolor="rgba(248,113,113,0.10)", line_width=0)
        ymax2 = max(float(np.max(r["spec"] * (1.0 / r["params"]["uA"] if cmp_norm else 1.0))) for r in runs)
        if cmp_scale == "log":
            fig2.update_yaxes(range=[math.log10(ymax2) - 5, math.log10(ymax2) + 0.4])
        fig2.update_layout(template="plotly_dark", height=480, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1324",
                           hovermode="x unified", margin=dict(l=60, r=20, t=20, b=50),
                           legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
                           xaxis=dict(title="Energia [keV]", range=[1, 26], gridcolor=C["grid"]),
                           yaxis=dict(title="ph/s/keV" + (" per µA" if cmp_norm else ""),
                                      type="log" if cmp_scale == "log" else "linear", gridcolor=C["grid"]))
        st.plotly_chart(fig2, width="stretch", config={"displaylogo": False})
        keys = sorted({k for r in runs for k in r["lines"]},
                      key=lambda k: -max(r["lines"].get(k, 0) for r in runs))[:20]
        keys = sorted(keys, key=lambda k: next(r["lines_E"][k] for r in runs if k in r["lines_E"]))
        rows = []
        for k in keys:
            row = {"Riga": k, "E [keV]": next(r["lines_E"][k] for r in runs if k in r["lines_E"])}
            ref = runs[0]["lines"].get(k, 0.0)
            for i, r in enumerate(runs):
                v = r["lines"].get(k, 0.0) / (r["params"]["uA"] if cmp_norm else 1.0)
                row[f"[{i+1}] cps"] = v
                if i > 0:
                    ref_v = ref / (runs[0]["params"]["uA"] if cmp_norm else 1.0)
                    row[f"[{i+1}]/[1]"] = v / ref_v if ref_v > 0 else np.nan
            rows.append(row)
        st.markdown("**Righe principali (rate rivelato)** — " +
                    " · ".join(f"[{i+1}] {r['label']}" for i, r in enumerate(runs)))
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.4g") for c in pd.DataFrame(rows).columns if c != "Riga"})

# ── TAB STRUMENTO ───────────────────────────────────────────────────────────
with tab_inst:
    c1, c2 = st.columns(2)
    # geometria
    cfgP = build_cfg(P)
    sx, sz, dx, dz = cfgP["src_x"], cfgP["src_z"], cfgP["det_x"], cfgP["det_z"]
    g = go.Figure()
    g.add_shape(type="rect", x0=-6, x1=6, y0=-0.6, y1=0, fillcolor="#3b3f4f", line_width=0)
    g.add_shape(type="rect", x0=-P["grid_hx"], x1=P["grid_hx"], y0=-0.06, y1=0.06,
                fillcolor="rgba(45,212,191,0.8)", line_width=0)
    for (x, z, col, name) in [(sx, sz, "#FBBF24", "tubo Mini-X2"), (dx, dz, "#F472B6", "X-123")]:
        g.add_trace(go.Scatter(x=[x, 0], y=[z, 0], mode="lines", line=dict(color=col, width=2, dash="dot"),
                               showlegend=False, hoverinfo="skip"))
        g.add_trace(go.Scatter(x=[x], y=[z], mode="markers+text", text=[name], textposition="top center",
                               marker=dict(size=16, color=col, symbol="square"), name=name))
    g.add_annotation(x=0, y=-0.35, text=f"area illuminata ±{P['grid_hx']:.2f} × ±{P['grid_hy']:.2f} cm",
                     showarrow=False, font=dict(color="#CFD6E6", size=11))
    g.update_layout(template="plotly_dark", height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1324",
                    margin=dict(l=40, r=10, t=40, b=40), showlegend=False,
                    title=dict(text=f"Geometria (piano x–z) · θ tubo {P['theta_src']:.1f}°, θ riv. {P['theta_det']:.1f}°, "
                                    f"scattering {psi:.0f}°", font=dict(size=13)),
                    xaxis=dict(title="x [cm]", range=[-7, 7], gridcolor=C["grid"]),
                    yaxis=dict(title="z [cm]", range=[-0.8, max(sz, dz) * 1.35], gridcolor=C["grid"],
                               scaleanchor="x", scaleratio=1))
    c1.plotly_chart(g, width="stretch", config={"displaylogo": False})

    # efficienze
    Eg = np.linspace(1.0, max(30.0, float(P["kv"])), 600)
    TBe = transmission_be(Eg, P["be_um"] / 1000.0)
    QE = qe_active_si(Eg, P["si_um"])
    mu_air = attenuation_coefficient(P["atmosphere"])
    L_out = max(P["D_det"] - P["det_vac"], 0.0)
    L_in = max(P["D_src"] - P["src_vac"], 0.0)
    Tair = np.exp(-mu_air(Eg) * L_out)
    e = go.Figure()
    for y, name, col, dash in [(TBe, f"Be {P['be_um']:.1f} µm", C["coh"], "dot"),
                               (QE, f"Si {P['si_um']:.0f} µm (fotoel.)", C["inc"], "dot"),
                               (Tair, f"{'aria' if P['atmosphere']=='air' else 'vuoto'} {L_out:.2f} cm", C["fluo"], "dot"),
                               (TBe * QE * Tair, "totale", C["total"], "solid")]:
        e.add_trace(go.Scatter(x=Eg, y=y, name=name, line=dict(color=col, dash=dash, width=2 if dash == "solid" else 1.4)))
    e.update_layout(template="plotly_dark", height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1324",
                    margin=dict(l=50, r=10, t=40, b=40), hovermode="x unified",
                    title=dict(text="Efficienza lato rivelatore", font=dict(size=13)),
                    legend=dict(orientation="h", y=-0.25), xaxis=dict(title="Energia [keV]", type="log", gridcolor=C["grid"]),
                    yaxis=dict(title="trasmissione / efficienza", range=[0, 1.02], gridcolor=C["grid"]))
    c2.plotly_chart(e, width="stretch", config={"displaylogo": False})

    c1, c2 = st.columns(2)
    try:
        cache_dir = Path(cfg0["_repo_root"]) / ".cache" / "tube"
        tcfg = dict(cfgP["tube"])
        tcfg["current_mA"] = P["uA"] / 1000.0
        with contextlib.redirect_stdout(io.StringIO()):
            Et, Pt = tube_source_term(tcfg, cache_dir)
        Tin = np.exp(-mu_air(Et) * L_in)
        t = go.Figure()
        t.add_trace(go.Scatter(x=Et, y=Pt, name="uscita tubo", line=dict(color=C["inc"], width=1.6)))
        if P["atmosphere"] == "air":
            t.add_trace(go.Scatter(x=Et, y=Pt * Tin, name=f"dopo {L_in:.2f} cm d'aria", line=dict(color=C["total"], width=1.6)))
        t.update_layout(template="plotly_dark", height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1324",
                        margin=dict(l=60, r=10, t=40, b=40), hovermode="x unified",
                        title=dict(text=f"Spettro del tubo (SpekPy) · {P['kv']} kV · Ag {P['target_um']} µm · {P['uA']} µA",
                                   font=dict(size=13)),
                        legend=dict(orientation="h", y=-0.25),
                        xaxis=dict(title="Energia [keV]", gridcolor=C["grid"]),
                        yaxis=dict(title="ph/s/keV/sr", type="log", gridcolor=C["grid"],
                                   range=[math.log10(Pt.max()) - 5, math.log10(Pt.max()) + 0.3]))
        c1.plotly_chart(t, width="stretch", config={"displaylogo": False})
    except Exception as exc:
        c1.warning(f"Spettro del tubo non disponibile: {exc}")

    comp = P["custom_comp"] or load_composition(Path(cfg0["paths"]["compositions"]) / f"{P['sample']}.yml")
    tot = sum(comp.values())
    cdf = pd.DataFrame({"El": list(comp), "w": [v / tot for v in comp.values()]})
    b = go.Figure(go.Bar(x=cdf["El"], y=cdf["w"] * 1e6, marker=dict(color=np.log10(cdf["w"]), colorscale="Tealrose"),
                         hovertemplate="%{x}: %{y:.4g} ppm<extra></extra>"))
    b.update_layout(template="plotly_dark", height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1324",
                    margin=dict(l=60, r=10, t=40, b=40),
                    title=dict(text=f"Composizione · {P['sample']}", font=dict(size=13)),
                    yaxis=dict(title="ppm (massa)", type="log", gridcolor=C["grid"]))
    c2.plotly_chart(b, width="stretch", config={"displaylogo": False})

    with st.expander("🧾 Config usata (YAML) e log del pipeline"):
        st.code(cfg_for_download(P), language="yaml")
        st.code(sim["log"] or "(nessun output: risultato dalla cache)", language="text")

# ── TAB NOTE ────────────────────────────────────────────────────────────────
with tab_notes:
    st.markdown(f"""
#### Cosa fa il pacchetto `xrf_model` e cosa aggiunge l'app
- **Fisica (`xrf_model`, invariato):** fluorescenza K primaria (Sherman/Shiraiwa–Fujino) e secondaria,
  righe L con Coster–Kronig e cascata K→L, Rayleigh con f0+f′, Compton Klein–Nishina × S con allargamento
  Doppler, assorbimento in aria, finestra Be ed efficienza del Si, risoluzione Fano + rumore. Tubo da SpekPy.
- **App:** la simulazione gira a 1 µA e viene scalata alla corrente scelta (il modello è lineare nella corrente:
  niente dead time né pile-up). Il **rumore di Poisson** è campionato qui sui canali MCA finali, dopo il rebin
  (`numpy.random.default_rng(seed).poisson`). P/B e LLD sono stime dell'app.

#### Limiti noti — leggili prima di confrontare con il misurato
1. **Argon non modellato.** L'aria assorbe ma non emette: mancano la riga Ar K (2.96 keV, ~3.5 % di Fe Kα nelle
   misure in aria) e lo scattering dell'aria. La zona 2.8–3.4 keV non può tornare.
2. **Nessun modello di footprint.** La griglia (±0.4 cm in config) è trattata come area illuminata uniformemente.
   Con il collimatore da 2 mm le intensità assolute escono circa ×4 [STIMA]; usa «Area illuminata → da collimatore»
   (formula di `lab_xrf.yml`: ½x = 0.443·D/cosθ, ½y = 0.443·D).
3. **Target Ag:** config 1.0 µm, scheda/manuale Amptek 0.75 ± 0.1 µm.
   Con 0.75 µm le righe calano del ~21–25 %.
4. **Si attivo:** nell'app il default è 500 µm (offerta ORTEC e manuale X-123), `lab_xrf.yml` ha {cfg0.get('si_active_layer_um')} µm.
5. **Matrice di risposta Geant4 non inclusa nel repo:** senza `.npz` i picchi sono gaussiani puri, senza escape
   (1.74 keV sotto ogni picco), coda e shelf. Anche il file di calibrazione MCA (`.calib`) non è nel repository.
6. **Sotto ~1 keV** lo spettro del tubo (SpekPy) non c'è: Na e Mg Kα non compaiono.
7. **Dead time** stimato con un modello paralizzabile e τ impostabile; **pile-up** e somme non modellati.
8. La geometria è nel piano x–z con il rivelatore puntato sullo spot (`det_normal = −r_det`), come in `lab_xrf.yml`.

#### Unità
- Tabella righe: *ingresso rivelatore* = ph/s che arrivano al rivelatore (geometria + aria), *rivelati* = × T_Be × QE_Si.
- Spettro: ph/s/keV (densità) oppure conteggi per canale nel live time scelto.
""")
    st.markdown("<span class='small-note'>GUI non ufficiale, non modifica il pacchetto `xrf_model`. "
                "Il risultato dipende dai parametri della config: i valori marcati STIMA sono ordini di grandezza.</span>",
                unsafe_allow_html=True)
