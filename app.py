"""
XRF sim — interfaccia grafica Streamlit
=======================================

GUI per il simulatore a parametri fondamentali (pacchetto `xrf_model` in ./src).
Il pacchetto non viene modificato: la config è costruita in memoria partendo da
configs/lab_xrf.yml e `run_one_grid` è chiamato così com'è. Le funzioni di
calcolo stanno in xrf_gui_core.py (importabili e testabili senza Streamlit).

Avvio:  streamlit run app.py   (dalla cartella del repo, accanto a src/ e configs/)
"""
from __future__ import annotations

import contextlib
import io
import math
import os
import secrets as pysecrets
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="XRF sim", page_icon="🌙", layout="wide", initial_sidebar_state="expanded")

try:
    import xrf_gui_core as core
except Exception as exc:   # pragma: no cover
    st.error("Non trovo `xrf_gui_core.py` o il pacchetto `xrf_model`. Metti `app.py` e `xrf_gui_core.py` "
             "nella cartella del repo (accanto a `src/` e `configs/`) e attiva il venv del repo.\n\n"
             f"Errore: {exc}")
    st.stop()


# ────────────────────────────────────────────────────────────────────────────
# Accesso con password (prima di qualsiasi calcolo)
# ────────────────────────────────────────────────────────────────────────────
def expected_password() -> str | None:
    try:
        v = st.secrets.get("app_password")
    except Exception:
        v = None
    return (str(v) if v else None) or os.environ.get("XRF_APP_PASSWORD") or None


_PW = expected_password()
if _PW and not st.session_state.get("auth_ok"):
    st.markdown("## 🔒 XRF sim")
    with st.form("login"):
        entered = st.text_input("Password", type="password")
        ok = st.form_submit_button("Entra", type="primary")
    if ok:
        if core.check_password(entered, _PW):
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Password errata.")
    st.stop()

C = dict(total="#2DD4BF", fluo="#F472B6", coh="#60A5FA", inc="#FBBF24", meas="#FB923C",
         poisson="rgba(229,233,242,0.55)", grid="rgba(148,163,184,0.12)")
RUN_COLORS = ["#2DD4BF", "#F472B6", "#FBBF24", "#60A5FA", "#A78BFA", "#34D399", "#F87171", "#E5E9F2"]
PLOT_BG = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1324")

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
# Cache (le chiavi includono l'impronta di lab_xrf.yml e dei file di composizione)
# ────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=8)
def base_config(stamp: str) -> dict:
    return core.base_config()


@st.cache_data(show_spinner=False, max_entries=48)
def simulate(key: str, p: dict, _base: dict) -> dict:
    return core.simulate_core(p, _base)


def compact(x, unit=""):
    for f, suf in ((1e9, " G"), (1e6, " M"), (1e3, " k")):
        if abs(x) >= f:
            return f"{x / f:.3g}{suf}{unit}"
    return f"{x:.0f} {unit}".strip()


cfg0 = base_config(core.file_stamp(core.CONFIG_PATH))
status = core.local_files_status(cfg0)
gdef = core.geometry_defaults(cfg0)
geo_param_ok, geo_reason = core.geometry_is_parametric(cfg0)
samples = core.list_samples(cfg0)
cfg_filters = {str(m): float(t) for m, t in (cfg0["tube"].get("filters") or {}).items()}
cfg_si = float(cfg0.get("si_active_layer_um", 450.0))

# ────────────────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────────────────
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
            "quelli della griglia di simulazione (30 eV), oppure quelli dello spettro misurato caricato. "
            "Il file `lab_xrf.yml` non viene toccato.")

# ────────────────────────────────────────────────────────────────────────────
# Sidebar: parametri fisici (form) + acquisizione
# ────────────────────────────────────────────────────────────────────────────
sb = st.sidebar
if _PW:
    if sb.button("Esci", icon="🔒"):
        st.session_state["auth_ok"] = False
        st.rerun()
else:
    sb.caption("Accesso senza password (uso locale). Per proteggerla: `app_password` in "
               "`.streamlit/secrets.toml` o variabile `XRF_APP_PASSWORD`.")
sb.markdown("### ⚙️ Parametri")

with sb.form("phys"):
    st.markdown("**Campione**")
    default_sample = cfg0["samples"][0] if cfg0["samples"][0] in samples else samples[0]
    sample = st.selectbox("Composizione (data/compositions)", samples, index=samples.index(default_sample))
    custom_file = st.file_uploader("…oppure carica un .yml (`composition:` → frazioni o wt%)",
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
    keep_filters = st.multiselect(
        "Filtri del config (tube.filters) da mantenere", list(cfg_filters), default=list(cfg_filters),
        format_func=lambda m: f"{m} {cfg_filters[m]:g} mm",
        help="Il Be è la finestra d'uscita del tubo: toglierlo non è fisico, ma resta possibile.")
    c1, c2 = st.columns(2)
    filter_mat = c1.selectbox("Filtro aggiuntivo", core.FILTER_MATERIALS, index=0,
                              help="Aggiunto ai filtri del config (stesso materiale: spessori sommati).")
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

    with st.expander("📐 Geometria", expanded=not geo_param_ok):
        geo_mode = st.radio("Modalità", ["config", "parametrica"], horizontal=True,
                            index=1 if geo_param_ok else 0,
                            format_func={"config": "usa geometria del config", "parametrica": "parametrica θ/D"}.get)
        if not geo_param_ok:
            st.warning(f"La geometria di lab_xrf.yml non è descrivibile con θ/D ({geo_reason}). "
                       "In modalità parametrica verrebbe sostituita.")
        c1, c2 = st.columns(2)
        theta_src = c1.number_input("θ tubo dalla normale [°]", 0.0, 85.0, round(gdef["theta_src"], 2), 0.5)
        D_src = c2.number_input("D tubo–spot [cm]", 0.3, 30.0, round(gdef["D_src"], 3), 0.05)
        theta_det = c1.number_input("θ rivelatore [°]", 0.0, 85.0, round(gdef["theta_det"], 2), 0.5)
        D_det = c2.number_input("D spot–cristallo [cm]", 0.3, 30.0, round(gdef["D_det"], 3), 0.05)
        src_vac = c1.number_input("Tratto in vuoto lato tubo [cm]", 0.0, 10.0,
                                  float(cfg0.get("src_vacuum_path_cm", 0.0)), 0.05)
        det_vac = c2.number_input("Tratto in vuoto lato rivelatore [cm]", 0.0, 10.0,
                                  float(cfg0.get("det_vacuum_path_cm", 0.0)), 0.05)
        st.caption("Parametrica: x = ∓D·sinθ, z = D·cosθ (formule di lab_xrf.yml), rivelatore puntato sullo spot. "
                   "θ/D si usano solo in modalità parametrica.")

    with st.expander("🔬 Rivelatore X-123 FAST-SDD"):
        c1, c2 = st.columns(2)
        det_area_mm2 = c1.number_input("Area attiva [mm²]", 1.0, 100.0, float(cfg0["det_area"]) * 100, 1.0,
                                       help="25 mm² nominali, 17 mm² collimati (config).")
        be_um = c2.number_input("Finestra Be [µm]", 0.0, 100.0, float(cfg0.get("be_window_mm", 0.0127)) * 1000, 0.1)
        si_um = c1.number_input("Si attivo [µm]", 100.0, 1000.0, 500.0, 10.0,
                                help="Default 500 µm (offerta ORTEC e manuale X-123).")
        c1.caption(f"lab_xrf.yml: {cfg_si:g} µm")
        fano = c2.number_input("Fano F", 0.05, 0.2, float(cfg0.get("detector_fano", 0.118)), 0.001, format="%.4f")
        w_ev = c1.number_input("w [eV]", 3.5, 3.8, float(cfg0.get("detector_w_eV", 3.65)), 0.01)
        noise_ev = c2.number_input("Rumore σ_el [eV rms]", 0.0, 60.0, float(cfg0.get("detector_noise_eV", 12.7)), 0.1)
        use_response = st.toggle("Usa matrice Geant4 se presente", value=True, disabled=not status["response"])

    if grid_n >= 6:
        st.warning(f"Griglia {grid_n}×{grid_n}: circa {1.0 + 0.25 * grid_n ** 2:.0f} s per run.")
    submitted = st.form_submit_button("▶ Simula", type="primary", width="stretch")

sb.markdown("### ⏱️ Acquisizione")
live_time = sb.number_input("Live time [s]", 1.0, 36000.0, 300.0, 10.0)
poisson_on = sb.toggle("Rumore Poisson", value=True)
c1, c2 = sb.columns([1, 1])
seed_mode = c1.radio("Seed", ["fisso", "casuale"], horizontal=False, label_visibility="collapsed")
if "seed_fixed" not in st.session_state:
    st.session_state["seed_fixed"] = 42
if "seed_rand" not in st.session_state:
    st.session_state["seed_rand"] = pysecrets.randbelow(10**9)
if seed_mode == "fisso":
    c2.number_input("Seed", 0, 10**9, key="seed_fixed", label_visibility="collapsed")
def _new_realization(mode):
    if mode == "fisso":
        st.session_state["seed_fixed"] = int(st.session_state["seed_fixed"]) + 1
    else:
        st.session_state["seed_rand"] = pysecrets.randbelow(10**9)


sb.button("🎲 Nuova realizzazione", width="stretch", disabled=not poisson_on,
          on_click=_new_realization, args=(seed_mode,))
seed = int(st.session_state["seed_fixed"] if seed_mode == "fisso" else st.session_state["seed_rand"])
sb.caption(f"Seed in uso: {seed}")
c1, c2 = sb.columns(2)
tpeak_us = c1.number_input("Peaking time [µs]", 0.05, 30.0, core.TPEAK_DEFAULT_us, 0.1,
                           help="STIMA: valore tipico; quello vero è TPEA= nel .mca (sezione DP5 CONFIGURATION).")
tau_ratio = c2.number_input("τ / Tpeak", 0.5, 5.0, core.TAU_OVER_TPEAK, 0.1,
                            help="Dalla scheda FAST SDD: Tpeak 9.6 µs → 18 kcps in uscita al 50 % di dead time; "
                                 "con modello paralizzabile τ ≈ 2.0 × Tpeak.")
tau_us = tpeak_us * tau_ratio

# composizione caricata
custom_comp, sample_name = None, sample
if custom_file is not None:
    try:
        tmpd = Path(tempfile.mkdtemp(prefix="xrfcomp_"))
        tf = tmpd / "custom.yml"
        tf.write_bytes(custom_file.getvalue())
        custom_comp = {k: float(v) for k, v in core.load_composition(tf).items()}
        sample_name = "custom_" + Path(custom_file.name).stem.replace(" ", "_")
        shutil.rmtree(tmpd, ignore_errors=True)
    except Exception as exc:
        sb.error(f"Composizione non valida: {exc}")

if grid_mode == "config":
    grid_hx, grid_hy = float(cfg0["grid_x_max"]), float(cfg0["grid_y_max"])
elif grid_mode == "collimatore":
    th = theta_src if geo_mode == "parametrica" else gdef["theta_src"]
    grid_hx = 0.443 * (beam_D_mm / 10.0) / math.cos(math.radians(th))
    grid_hy = 0.443 * (beam_D_mm / 10.0)
else:
    grid_hx, grid_hy = hx_man, hy_man

params = dict(sample=sample_name, custom_comp=custom_comp, secondary=secondary, kv=kv, uA=uA,
              target_um=target_um, keep_filters=sorted(keep_filters), filter_mat=filter_mat, filter_mm=filter_mm,
              geo_mode=geo_mode, theta_src=theta_src, D_src=D_src, theta_det=theta_det, D_det=D_det,
              det_area_mm2=det_area_mm2, atmosphere=atmosphere, src_vac=src_vac, det_vac=det_vac,
              grid_hx=round(grid_hx, 5), grid_hy=round(grid_hy, 5), grid_n=grid_n, be_um=be_um, si_um=si_um,
              fano=fano, w_ev=w_ev, noise_ev=noise_ev, use_response=use_response)
if submitted or "params" not in st.session_state:
    st.session_state["params"] = params
P = dict(st.session_state["params"])
P["live_time"] = live_time

if abs(P["si_um"] - cfg_si) > 0.5:
    sb.caption(f"⚠️ Si attivo {P['si_um']:g} µm ≠ lab_xrf.yml ({cfg_si:g} µm)"
               + (" — con la matrice Geant4 lo spessore deve essere quello della simulazione MC." if status["response"] else ""))

# ────────────────────────────────────────────────────────────────────────────
# Simulazione
# ────────────────────────────────────────────────────────────────────────────
power_W = P["kv"] * P["uA"] * 1e-3
try:
    with st.spinner(f"Simulazione FP in corso — {P['sample']}, {P['kv']} kV, griglia {P['grid_n']}×{P['grid_n']}…"):
        sim1 = simulate(core.phys_key(P, cfg0), {k: P[k] for k in core.PHYS_KEYS}, cfg0)
except Exception as exc:
    st.error("La simulazione è fallita.")
    st.exception(exc)
    st.stop()
sim = core.scaled(sim1, P["uA"])
cfgP = sim["cfg"]

with sb.expander("🧪 Composizione del campione simulato", expanded=False):
    cdf = pd.DataFrame({"El": list(sim["comp"]), "wt %": [100 * v for v in sim["comp"].values()]})
    cdf["ppm"] = cdf["wt %"] * 1e4
    st.dataframe(cdf.sort_values("wt %", ascending=False), hide_index=True, width="stretch",
                 column_config={"wt %": st.column_config.NumberColumn(format="%.3f"),
                                "ppm": st.column_config.NumberColumn(format="%.0f")})
    st.caption("Frazioni normalizzate a somma 1 (usate anche per tabelle e LLD).")

# ── spettro misurato ────────────────────────────────────────────────────────
meas = None
with st.expander("📥 Spettro misurato (.mca Amptek/PMCA) — sovrapponi e confronta", expanded=False):
    up = st.file_uploader("File .mca", type=["mca", "MCA", "txt"], key="meas")
    if up is not None:
        try:
            meas = core.parse_mca(up.getvalue())
        except core.McaError as exc:
            st.error(f"File non valido: {exc}")
        except Exception as exc:
            st.error(f"Errore nella lettura del file ({type(exc).__name__}): {exc}")
    if meas is not None:
        n = len(meas["counts"])
        if meas["cal_note"]:
            st.warning(f"Calibrazione: {meas['cal_note']}. Inserisci A, B a mano.")
        elif meas["calib"] is None:
            st.warning("Nessuna sezione <<CALIBRATION>>: inserisci A, B a mano.")
        cal = meas["calib"] or (0.0, 0.006, 0.0)
        c1, c2, c3, c4 = st.columns(4)
        A = c1.number_input("A [keV]", -5.0, 5.0, float(cal[0]), 0.001, format="%.5f")
        B = c2.number_input("B [keV/canale]", core.B_RANGE_keV[0], core.B_RANGE_keV[1], float(cal[1]), 0.0001, format="%.6f")
        Cq = c3.number_input("C [keV/canale²]", -1e-5, 1e-5, float(cal[2]), 1e-9, format="%.3e")
        if meas["live_missing"]:
            st.warning("LIVE_TIME mancante o nullo nel file: inserisci il live time a mano "
                       + (f"(REAL_TIME = {meas['real']:g} s)." if meas["real"] else "."))
        lt_m = c4.number_input("Live time misurato [s]", 0.001, 1e6, float(meas["live"] or meas["real"] or 1.0), 1.0)
        meas.update(A=A, B=B, Cq=Cq, live=lt_m)
        c1, c2 = st.columns(2)
        meas["align"] = c1.toggle("Simula sui canali del misurato", value=True)
        meas["norm"] = c2.radio("Normalizzazione", ["nessuna", "live time", "su un picco"], index=1, horizontal=True)
        cal_txt = (f"dal file ({meas['cal_label'] or 'senza etichetta'}, unità {meas['cal_unit']}, "
                   f"{'quadratica' if meas['cal_deg'] == 2 else 'lineare'})" if meas["calib"] else "manuale")
        st.caption(f"{n} canali · LIVE_TIME {meas['live']:g} s · REAL_TIME {meas['real'] or '—'} s · "
                   f"calibrazione {cal_txt} · conteggi totali {meas['counts'].sum():,.0f}")
        info = {"Peaking time TPEA [µs]": meas["tpeak_us"], "Gain (GAIN)": meas["gain"],
                "Fast count": meas["fast"], "Slow count": meas["slow"],
                "Dead time misurato [%]": None if meas["meas_dt"] is None else round(100 * meas["meas_dt"], 2)}
        info = {k: v for k, v in info.items() if v is not None}
        if info or meas["dp5"] or meas["status"]:
            with st.popover("ℹ️ Impostazioni DP5 e stato DPP dal file"):
                if info:
                    st.dataframe(pd.DataFrame({"voce": list(info), "valore": [str(v) for v in info.values()]}),
                                 hide_index=True, width="stretch")
                if meas["dp5"]:
                    st.markdown("**DP5 CONFIGURATION**")
                    st.dataframe(pd.DataFrame({"chiave": list(meas["dp5"]), "valore": list(meas["dp5"].values())}),
                                 hide_index=True, width="stretch", height=240)
                if meas["status"]:
                    st.markdown("**DPP STATUS**")
                    st.dataframe(pd.DataFrame({"voce": list(meas["status"]), "valore": list(meas["status"].values())}),
                                 hide_index=True, width="stretch", height=240)

# ── canali MCA ──────────────────────────────────────────────────────────────
if meas is not None and meas.get("align"):
    calib, n_ch, ch_src = (meas["A"], meas["B"], meas["Cq"]), len(meas["counts"]), "canali del misurato"
elif status["calibration"]:
    calib, n_ch, ch_src = core.read_calibration(cfg0["mca_calibration"]), int(cfg0["mca_channels"]), "calibrazione MCA della config"
else:
    step = float(np.mean(np.diff(sim["sE"])))
    calib, n_ch = (float(sim["sE"][0]), step, 0.0), len(sim["sE"])
    ch_src = f"griglia di simulazione ({step*1000:.0f} eV/canale)"
edges = core.edges_from_calib(calib, n_ch)
ch_E = 0.5 * (edges[1:] + edges[:-1])
ch_w = np.diff(edges)
rate_ch = core.rate_on_channels(sim, edges)
expected = rate_ch * live_time
noisy = core.poisson_counts(expected, seed)
counts = noisy if poisson_on else np.round(expected)

total_rate = float(np.trapezoid(sim["spec"], sim["sE"]))
dead = core.dead_time_fraction(total_rate, tau_us)
real_time_est = live_time / max(1.0 - dead, 1e-6)
fwhm59 = 2.3548 * float(core.sigma_kev(5.895, P["fano"], P["noise_ev"], P["w_ev"])) * 1000

m = st.columns(6)
m[0].metric("Conteggi totali", compact(float(counts.sum())),
            help=f"Nel live time di {live_time:g} s, {'con' if poisson_on else 'senza'} Poisson")
m[1].metric("Rate rivelato", compact(total_rate, "cps"), help="Integrale dello spettro rivelato (sopra ~1 keV)")
m[2].metric("Dead time (STIMA)", f"{100*dead:.1f} %",
            help=f"1−exp(−R·τ), τ = {tau_us:.2g} µs ({tau_ratio:g} × Tpeak {tpeak_us:g} µs); "
                 f"real time stimato {real_time_est:.0f} s. Ar, righe della box e pile-up non inclusi.")
m[3].metric("Potenza tubo", f"{power_W:.2f} W", delta=f"max {core.MINI_X2_MAX_W:.0f} W",
            delta_color="inverse" if power_W > core.MINI_X2_MAX_W else "off")
m[4].metric("FWHM @ 5.9 keV", f"{fwhm59:.0f} eV")
m[5].metric("Tempo di calcolo", f"{sim['dt']:.1f} s", help="Primo run per kV/filtro/target: +~7 s per SpekPy. Poi cache.")
if power_W > core.MINI_X2_MAX_W:
    st.error(f"⚠️ {P['kv']} kV × {P['uA']} µA = {power_W:.2f} W: oltre i 4 W del Mini-X2. "
             f"A {P['kv']} kV la corrente massima è {core.MINI_X2_MAX_W / P['kv'] * 1000:.0f} µA.")
if sim["n_valid"] < P["grid_n"] ** 2:
    st.warning(f"Solo {sim['n_valid']} celle su {P['grid_n']**2} sono viste da tubo e rivelatore.")
if meas is not None and meas.get("meas_dt") is not None:
    st.caption(f"Dead time misurato dal file: {100*meas['meas_dt']:.1f} % "
               f"({'fast/slow count' if meas['fast'] else '1 − LIVE/REAL'}) · stimato per la simulazione: {100*dead:.1f} %")

tab_spec, tab_lines, tab_cmp, tab_inst, tab_notes = st.tabs(
    ["📈 Spettro", "📋 Righe & LLD", "🔀 Confronto run", "🛰️ Strumento", "📝 Note fisiche"])
psi = core.scattering_angle_deg(cfgP)
emax_default = float(min(P["kv"], 40.0))

# ── TAB SPETTRO ─────────────────────────────────────────────────────────────
with tab_spec:
    c = st.columns([1.2, 1.2, 2.2, 1.4])
    ymode = c[0].radio("Asse y", ["conteggi/canale", "ph/s/keV"])
    yscale = c[1].radio("Scala", ["log", "lin"])
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

    def as_y(ch_counts):          # conteggi per canale → asse scelto
        return ch_counts if to_counts else ch_counts / (live_time * ch_w)

    has_meas = meas is not None
    fig = make_subplots(rows=2 if has_meas else 1, cols=1, shared_xaxes=True,
                        row_heights=[0.76, 0.24] if has_meas else [1.0], vertical_spacing=0.04)
    if poisson_on:
        fig.add_trace(go.Scatter(x=ch_E, y=as_y(counts), mode="lines", line=dict(color=C["poisson"], width=1, shape="hvh"),
                                 name=f"simulato + Poisson (seed {seed})"), 1, 1)
    if show_expected or not poisson_on:
        fig.add_trace(go.Scatter(x=ch_E, y=as_y(expected), mode="lines", line=dict(color=C["total"], width=2),
                                 name="simulato (atteso)" + (" · Geant4" if sim["used_response"] else "")), 1, 1)
    comp_map = {"fluorescenza": ("comp_fluo", C["fluo"]), "coerente": ("comp_coh", C["coh"]),
                "incoerente": ("comp_inc", C["inc"])}
    for name in show_comp:
        key, col = comp_map[name]
        fig.add_trace(go.Scatter(x=ch_E, y=as_y(core.component_on_channels(sim, key, edges) * live_time),
                                 mode="lines", line=dict(color=col, width=1.3, dash="dot"), name=name), 1, 1)

    k_meas = 1.0
    if has_meas:
        m_edges = core.edges_from_calib((meas["A"], meas["B"], meas["Cq"]), len(meas["counts"]))
        m_E = 0.5 * (m_edges[1:] + m_edges[:-1])
        m_w = np.diff(m_edges)
        if meas["norm"] == "live time":
            k_meas = live_time / meas["live"]
        elif meas["norm"] == "su un picco":
            gl = core.grouped_lines(sim["fluo"])
            gl = gl[(gl["E"] > max(erange[0], 1.0)) & (gl["E"] < erange[1])].sort_values("cps", ascending=False)
            opts = [f"{r.Element} {r.Group} ({r.E:.2f} keV)" for r in gl.itertuples()]
            if opts:
                choice = st.selectbox("Picco per la normalizzazione", opts)
                E0 = float(choice.split("(")[1].split()[0])
                fw = 2.3548 * float(core.sigma_kev(E0, P["fano"], P["noise_ev"], P["w_ev"]))
                win = np.array([E0 - 0.9 * fw, E0 + 0.9 * fw])
                sim_w = float(core.rate_on_channels(sim, win)[0] * live_time)
                sel = (m_E >= win[0]) & (m_E <= win[1])
                meas_w = float(meas["counts"][sel].sum())
                k_meas = sim_w / meas_w if meas_w > 0 else 1.0
        y_m = meas["counts"] * k_meas if to_counts else meas["counts"] * k_meas / (live_time * m_w)
        fig.add_trace(go.Scatter(x=m_E, y=y_m, mode="lines", line=dict(color=C["meas"], width=1.2, shape="hvh"),
                                 name=f"misurato × {k_meas:.3g}"), 1, 1)
        if meas.get("align"):
            res = core.residuals(meas["counts"], expected, k_meas)
            fig.add_trace(go.Scatter(x=ch_E, y=res, mode="lines", line=dict(color=C["meas"], width=1),
                                     name="(mis − sim)/√sim", showlegend=False), 2, 1)
            fig.update_yaxes(title_text="residui [σ]", row=2, col=1, gridcolor=C["grid"], zeroline=True,
                             zerolinecolor="rgba(229,233,242,0.4)")
        else:
            fig.add_annotation(text="Residui disponibili con «Simula sui canali del misurato»", row=2, col=1,
                               x=0.5, y=0.5, xref="x domain", yref="y2 domain", showarrow=False,
                               font=dict(color="#8C96B0"))

    if show_ar:
        fig.add_vrect(x0=core.AR_ZONE[0], x1=core.AR_ZONE[1], fillcolor="rgba(248,113,113,0.10)", line_width=0,
                      annotation_text="Ar K (2.96 keV): non modellato", annotation_position="top left",
                      annotation_font=dict(color="#FCA5A5", size=11), row=1, col=1)

    ref_x, ref_y = ch_E, as_y(expected)
    if show_labels:
        gl = core.grouped_lines(sim["fluo"])
        gl = gl[(gl["E"] >= erange[0]) & (gl["E"] <= erange[1]) & (gl["E"] >= ref_x[0])]
        gl = gl.sort_values("cps", ascending=False).head(n_labels)
        items = [(f"{r.Element} {r.Group}", r.E) for r in gl.itertuples()]
        if P["kv"] > core.AG_K_EDGE_keV:
            scat = [("Ag Kα · Rayleigh", 22.10), ("Ag Kβ · Rayleigh", 24.94),
                    ("Ag Kα · Compton", core.compton_energy(22.10, psi))]
            items += [(n_, e) for n_, e in scat if erange[0] <= e <= erange[1]]
        for label, E0 in items:
            j = np.searchsorted(ref_x, E0)
            lo_j, hi_j = max(j - 3, 0), min(j + 4, len(ref_y))
            yv = float(ref_y[lo_j:hi_j].max()) if hi_j > lo_j else 0.0
            if yv <= 0:
                continue
            ya = math.log10(yv) if yscale == "log" else yv
            fig.add_annotation(x=E0, y=ya, text=label, showarrow=True, arrowhead=0, arrowwidth=1,
                               arrowcolor="rgba(229,233,242,0.35)", ax=0, ay=-26, textangle=-60,
                               font=dict(size=10, color="#93C5FD" if "Ag" in label else "#E5E9F2"), row=1, col=1)

    fig.update_yaxes(type="log" if yscale == "log" else "linear",
                     title_text="conteggi / canale" if to_counts else "ph / s / keV", gridcolor=C["grid"], row=1, col=1)
    if yscale == "log":
        sel = (ref_x >= erange[0]) & (ref_x <= erange[1])
        ymax = float(np.max(ref_y[sel])) if sel.any() else 1.0
        floor = 0.5 if to_counts else ymax * 1e-6
        fig.update_yaxes(range=[math.log10(max(floor, 1e-12)), math.log10(max(ymax, 1e-9) * 6)], row=1, col=1)
    fig.update_xaxes(range=list(erange), gridcolor=C["grid"], row=1, col=1)
    fig.update_xaxes(title_text="Energia [keV]", gridcolor=C["grid"], row=2 if has_meas else 1, col=1)
    fig.update_layout(**PLOT_BG, height=620 if has_meas else 560, margin=dict(l=60, r=20, t=40, b=50),
                      hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
                      title=dict(text=f"<b>{P['sample']}</b> · {P['kv']} kV · {P['uA']} µA · "
                                      f"{'aria' if P['atmosphere']=='air' else 'vuoto'} · LT {live_time:g} s · "
                                      f"griglia {P['grid_n']}×{P['grid_n']}", font=dict(size=14), x=0.0, y=0.995))
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "toImageButtonOptions":
                    {"format": "png", "filename": f"xrf_{P['sample']}_{P['kv']}kV", "scale": 2}})
    st.markdown(f"<span class='small-note'>Canali: {ch_src} · {n_ch} canali · picchi integrati sui canali "
                f"(continuo con interpolazione monotona della cumulata) · angolo di scattering {psi:.0f}° · "
                f"spettro del tubo da {sim['E'][0]:.3f} keV.</span>", unsafe_allow_html=True)

    st.markdown("#### 💾 Salva ed esporta")
    c = st.columns([2, 1, 1, 1, 1, 1])
    filt_txt = ", ".join(f"{m_} {t:g} mm" for m_, t in cfgP["tube"]["filters"].items())
    default_label = (f"{P['sample']} · {P['kv']} kV {P['uA']} µA · {'aria' if P['atmosphere']=='air' else 'vuoto'}"
                     + (f" · {P['filter_mat']} {P['filter_mm']} mm" if P['filter_mat'] != 'nessuno' and P['filter_mm'] > 0 else ""))
    run_label = c[0].text_input("Etichetta run", default_label, label_visibility="collapsed")
    if c[1].button("📌 Aggiungi al confronto", width="stretch"):
        runs = st.session_state.setdefault("runs", [])
        gl = core.grouped_lines(sim["fluo"])
        runs.append(dict(label=run_label, E=sim["sE"], spec=sim["spec"], params=dict(P),
                         lines={f"{r.Element} {r.Group}": r.cps for r in gl.itertuples()},
                         lines_E={f"{r.Element} {r.Group}": r.E for r in gl.itertuples()}))
        st.session_state["runs"] = runs[-8:]
        st.toast(f"Run salvato: {run_label}", icon="📌")

    base = f"xrf_{P['sample']}_{P['kv']}kV_{P['uA']}uA_{P['atmosphere']}"
    desc = (f"XRF sim {P['sample']} {P['kv']}kV {P['uA']}uA {P['atmosphere']} filtri[{filt_txt}] "
            f"{'Poisson seed ' + str(seed) if poisson_on else 'atteso'} DT_stima {100*dead:.1f}%")
    c[2].download_button("⬇ .mca", core.write_mca(counts, calib, live_time, real_time_est, desc),
                         file_name=base + ".mca", width="stretch",
                         help=f"REAL_TIME = LIVE_TIME/(1−DT) = {real_time_est:.1f} s (dead time stimato)")
    spec_df = pd.DataFrame({"Energy_keV": sim["E"], "fluorescence_ph_s_keV": sim["comp_fluo"],
                            "coherent_ph_s_keV": sim["comp_coh"], "incoherent_ph_s_keV": sim["comp_inc"]})
    spec_df["total_gaussian_ph_s_keV"] = spec_df.iloc[:, 1:].sum(axis=1)
    chan_df = pd.DataFrame({"channel": np.arange(n_ch), "E_keV": ch_E, "expected_counts": expected, "counts": counts})
    csv_buf = io.StringIO()
    csv_buf.write(f"# {desc}; live_time_s={live_time}; canali: {ch_src}\n")
    chan_df.to_csv(csv_buf, index=False)
    csv_buf.write("\n# Densità sulla griglia di simulazione (ph/s/keV, risposta gaussiana)\n")
    spec_df.to_csv(csv_buf, index=False)
    c[3].download_button("⬇ .csv", csv_buf.getvalue().encode(), file_name=base + ".csv", width="stretch")
    lld_def_dl = st.session_state.get("lld_def", "3√B")
    xbuf = io.BytesIO()
    lines_out = sim["fluo"].copy()
    lines_out["counts_in_live_time"] = lines_out["cps_det"] * live_time
    with pd.ExcelWriter(xbuf) as xw:
        lines_out.to_excel(xw, sheet_name="Righe", index=False)
        core.lld_table(sim, live_time, float(sim["E"][0]), float(P["kv"]), lld_def_dl).to_excel(
            xw, sheet_name="LLD (stima)", index=False)
        pd.DataFrame({"parametro": list(P), "valore": [str(v) for v in P.values()]}).to_excel(
            xw, sheet_name="Parametri", index=False)
    c[4].download_button("⬇ .xlsx", xbuf.getvalue(), file_name=base + "_righe.xlsx", width="stretch")
    cfg_text, cfg_extra = core.export_config(P, cfg0, P["uA"], live_time)
    if cfg_extra:
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(base + ".yml", cfg_text)
            for name_, txt in cfg_extra.items():
                z.writestr(name_, txt)
        c[5].download_button("⬇ config", zbuf.getvalue(), file_name=base + "_config.zip", width="stretch",
                             help="Config + composizione caricata (compositions_gui/).")
    else:
        c[5].download_button("⬇ config", cfg_text.encode(), file_name=base + ".yml", width="stretch")

# ── TAB RIGHE ───────────────────────────────────────────────────────────────
with tab_lines:
    if sim["used_response"]:
        st.warning("Matrice Geant4 attiva: lo spettro usa la matrice, ma tabella righe, P/B e LLD usano picchi "
                   "gaussiani e QE solo fotoelettrica (approssimazione).")
    st.markdown("#### Righe di fluorescenza")
    df = sim["fluo"][sim["fluo"]["Intensity_ph_s"] > 0]
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

    c1, c2 = st.columns([3, 2])
    c1.markdown("#### Picco/fondo e limite di rivelabilità  <span class='badge warn'>STIMA</span>", unsafe_allow_html=True)
    lld_def = c2.radio("Definizione LLD", ["3√B", "Currie"], horizontal=True, key="lld_def",
                       help="3√B: 3√B/N·c. Currie: L_D ≈ 2.71 + 4.65√B conteggi (blank appaiato).")
    lt = core.lld_table(sim, live_time, float(sim["E"][0]), float(P["kv"]), lld_def)
    lcol = f"LLD {lld_def} [ppm] (STIMA)"
    if not lt.empty:
        c1, c2 = st.columns([3, 2])
        fmt = {k: st.column_config.NumberColumn(format=f) for k, f in [
            ("Conc. [ppm]", "%.1f"), ("FWHM [eV]", "%.0f"), ("N picco (finestra)", "%.3g"),
            ("B fondo (finestra)", "%.3g"), ("P/B", "%.3g"), ("σ statistico [%]", "%.2f"),
            ("LLD Kα [ppm]", "%.3g"), ("LLD Lα [ppm]", "%.3g"), (lcol, "%.3g")]}
        c1.dataframe(lt, hide_index=True, width="stretch", height=420, column_config=fmt)
        lf = go.Figure(go.Bar(x=lt["Elemento"] + " " + lt["Riga usata"], y=lt[lcol],
                              marker=dict(color=lt["E [keV]"], colorscale="Tealrose", showscale=False),
                              hovertemplate="%{x}: %{y:.3g} ppm<extra></extra>"))
        lf.update_layout(**PLOT_BG, height=420, yaxis=dict(type="log", title="LLD [ppm]", gridcolor=C["grid"]),
                         margin=dict(l=50, r=10, t=30, b=40),
                         title=dict(text=f"LLD ({lld_def}) in {live_time:g} s", font=dict(size=13)))
        c2.plotly_chart(lf, width="stretch", config={"displaylogo": False})
    st.caption(f"Per ogni elemento si valutano Kα e Lα (se eccitate e dentro la griglia) e si sceglie la LLD migliore. "
               f"Finestra = {core.WIN_FWHM} × FWHM; N = conteggi netti del gruppo nella finestra; B = continuo di "
               "scattering + code delle altre righe nella finestra. Concentrazioni normalizzate a somma 1, risposta "
               "lineare. Con 3√B e B < 10 conteggi la LLD non è definita (cella vuota). Senza Ar, pile-up, escape e "
               "shelf: ordine di grandezza, non un valore di specifica.")

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
            fig2.add_vrect(x0=core.AR_ZONE[0], x1=core.AR_ZONE[1], fillcolor="rgba(248,113,113,0.10)", line_width=0)
        ymax2 = max(float(np.max(r["spec"] * (1.0 / r["params"]["uA"] if cmp_norm else 1.0))) for r in runs)
        if cmp_scale == "log":
            fig2.update_yaxes(range=[math.log10(ymax2) - 5, math.log10(ymax2) + 0.4])
        fig2.update_layout(**PLOT_BG, height=480, hovermode="x unified", margin=dict(l=60, r=20, t=20, b=50),
                           legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
                           xaxis=dict(title="Energia [keV]", range=[1, 26], gridcolor=C["grid"]),
                           yaxis=dict(title="ph/s/keV" + (" per µA" if cmp_norm else ""),
                                      type="log" if cmp_scale == "log" else "linear", gridcolor=C["grid"]))
        st.plotly_chart(fig2, width="stretch", config={"displaylogo": False})
        keys = sorted({k for r in runs for k in r["lines"]}, key=lambda k: -max(r["lines"].get(k, 0) for r in runs))[:20]
        keys = sorted(keys, key=lambda k: next(r["lines_E"][k] for r in runs if k in r["lines_E"]))
        rows = []
        for k in keys:
            row = {"Riga": k, "E [keV]": next(r["lines_E"][k] for r in runs if k in r["lines_E"])}
            ref_v = runs[0]["lines"].get(k, 0.0) / (runs[0]["params"]["uA"] if cmp_norm else 1.0)
            for i, r in enumerate(runs):
                v = r["lines"].get(k, 0.0) / (r["params"]["uA"] if cmp_norm else 1.0)
                row[f"[{i+1}] cps"] = v
                if i > 0:
                    row[f"[{i+1}]/[1]"] = v / ref_v if ref_v > 0 else np.nan
            rows.append(row)
        st.markdown("**Righe principali (rate rivelato)** — " + " · ".join(f"[{i+1}] {r['label']}" for i, r in enumerate(runs)))
        rdf = pd.DataFrame(rows)
        st.dataframe(rdf, hide_index=True, width="stretch",
                     column_config={c_: st.column_config.NumberColumn(format="%.4g") for c_ in rdf.columns if c_ != "Riga"})

# ── TAB STRUMENTO ───────────────────────────────────────────────────────────
with tab_inst:
    c1, c2 = st.columns(2)
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
    off_plane = abs(cfgP["src_y"]) > 1e-6 or abs(cfgP["det_y"]) > 1e-6
    g.update_layout(**PLOT_BG, height=360, margin=dict(l=40, r=10, t=40, b=40), showlegend=False,
                    title=dict(text=f"Geometria (proiezione x–z{' — componenti y non mostrate' if off_plane else ''}) · "
                                    f"scattering {psi:.0f}° · {'config' if P['geo_mode']=='config' else 'parametrica'}",
                               font=dict(size=13)),
                    xaxis=dict(title="x [cm]", range=[-7, 7], gridcolor=C["grid"]),
                    yaxis=dict(title="z [cm]", range=[-0.8, max(sz, dz) * 1.35], gridcolor=C["grid"], scaleanchor="x", scaleratio=1))
    c1.plotly_chart(g, width="stretch", config={"displaylogo": False})

    r_src = math.dist((0, 0, 0), (cfgP["src_x"], cfgP["src_y"], cfgP["src_z"]))
    r_det = math.dist((0, 0, 0), (cfgP["det_x"], cfgP["det_y"], cfgP["det_z"]))
    Eg = np.linspace(1.0, max(30.0, float(P["kv"])), 600)
    TBe = core.transmission_be(Eg, P["be_um"] / 1000.0)
    QE = core.qe_active_si(Eg, P["si_um"])
    mu_air = core.attenuation_coefficient(P["atmosphere"])
    L_out = max(r_det - P["det_vac"], 0.0)
    L_in = max(r_src - P["src_vac"], 0.0)
    Tair = np.exp(-mu_air(Eg) * L_out)
    e = go.Figure()
    for y, name, col, dash in [(TBe, f"Be {P['be_um']:.1f} µm", C["coh"], "dot"),
                               (QE, f"Si {P['si_um']:.0f} µm (fotoel.)", C["inc"], "dot"),
                               (Tair, f"{'aria' if P['atmosphere']=='air' else 'vuoto'} {L_out:.2f} cm", C["fluo"], "dot"),
                               (TBe * QE * Tair, "totale", C["total"], "solid")]:
        e.add_trace(go.Scatter(x=Eg, y=y, name=name, line=dict(color=col, dash=dash, width=2 if dash == "solid" else 1.4)))
    e.update_layout(**PLOT_BG, height=360, margin=dict(l=50, r=10, t=40, b=40), hovermode="x unified",
                    title=dict(text="Efficienza lato rivelatore", font=dict(size=13)),
                    legend=dict(orientation="h", y=-0.25), xaxis=dict(title="Energia [keV]", type="log", gridcolor=C["grid"]),
                    yaxis=dict(title="trasmissione / efficienza", range=[0, 1.02], gridcolor=C["grid"]))
    c2.plotly_chart(e, width="stretch", config={"displaylogo": False})

    c1, c2 = st.columns(2)
    try:
        tcfg = dict(cfgP["tube"])
        tcfg["current_mA"] = P["uA"] / 1000.0
        with contextlib.redirect_stdout(io.StringIO()):
            Et, Pt = core.tube_source_term(tcfg, Path(cfg0["_repo_root"]) / ".cache" / "tube")
        t = go.Figure()
        t.add_trace(go.Scatter(x=Et, y=Pt, name="uscita tubo", line=dict(color=C["inc"], width=1.6)))
        if P["atmosphere"] == "air":
            t.add_trace(go.Scatter(x=Et, y=Pt * np.exp(-mu_air(Et) * L_in), name=f"dopo {L_in:.2f} cm d'aria",
                                   line=dict(color=C["total"], width=1.6)))
        t.update_layout(**PLOT_BG, height=360, margin=dict(l=60, r=10, t=40, b=40), hovermode="x unified",
                        title=dict(text=f"Spettro del tubo (SpekPy) · {P['kv']} kV · Ag {P['target_um']} µm · "
                                        f"filtri {filt_txt or 'nessuno'}", font=dict(size=13)),
                        legend=dict(orientation="h", y=-0.25), xaxis=dict(title="Energia [keV]", gridcolor=C["grid"]),
                        yaxis=dict(title="ph/s/keV/sr", type="log", gridcolor=C["grid"],
                                   range=[math.log10(Pt.max()) - 5, math.log10(Pt.max()) + 0.3]))
        c1.plotly_chart(t, width="stretch", config={"displaylogo": False})
    except Exception as exc:
        c1.warning(f"Spettro del tubo non disponibile: {exc}")

    cdf = pd.DataFrame({"El": list(sim["comp"]), "w": list(sim["comp"].values())})
    b = go.Figure(go.Bar(x=cdf["El"], y=cdf["w"] * 1e6, marker=dict(color=np.log10(cdf["w"].clip(lower=1e-12)), colorscale="Tealrose"),
                         hovertemplate="%{x}: %{y:.4g} ppm<extra></extra>"))
    b.update_layout(**PLOT_BG, height=360, margin=dict(l=60, r=10, t=40, b=40),
                    title=dict(text=f"Composizione normalizzata · {P['sample']}", font=dict(size=13)),
                    yaxis=dict(title="ppm (massa)", type="log", gridcolor=C["grid"]))
    c2.plotly_chart(b, width="stretch", config={"displaylogo": False})

    with st.expander("🧾 Config usata (YAML) e log del pipeline"):
        st.code(cfg_text, language="yaml")
        st.code(sim["log"] or "(nessun output: risultato dalla cache)", language="text")

# ── TAB NOTE ────────────────────────────────────────────────────────────────
with tab_notes:
    st.markdown(f"""
#### Cosa fa il pacchetto `xrf_model` e cosa aggiunge l'app
- **Fisica (`xrf_model`, invariato):** fluorescenza K primaria (Sherman/Shiraiwa–Fujino) e secondaria,
  righe L con Coster–Kronig e cascata K→L, Rayleigh con f0+f′, Compton Klein–Nishina × S con allargamento
  Doppler, assorbimento in aria, finestra Be ed efficienza del Si, risoluzione Fano + rumore. Tubo da SpekPy.
- **App:** la simulazione gira a 1 µA e viene scalata alla corrente scelta (il modello è lineare nella corrente:
  niente dead time né pile-up). I picchi gaussiani sono integrati analiticamente sui canali MCA di destinazione;
  il continuo è ridistribuito con un'interpolazione monotona della cumulata (totale conservato). Il **rumore di
  Poisson** è campionato sui canali finali (`numpy.random.default_rng(seed).poisson`). P/B, LLD e dead time sono stime.

#### Limiti noti
1. **Argon non modellato.** L'aria assorbe ma non emette né diffonde: la riga Ar K (2.96 keV) e lo scattering
   dell'aria mancano, quindi la zona 2.8–3.4 keV non è confrontabile con una misura in aria.
2. **Nessun modello di footprint.** La griglia (±0.4 cm in config) è trattata come area illuminata uniformemente.
   Con un collimatore le intensità assolute sono sovrastimate; «Area illuminata → da collimatore» usa la formula
   di `lab_xrf.yml` (½x = 0.443·D/cosθ, ½y = 0.443·D).
3. **Target Ag:** config 1.0 µm, scheda/manuale Amptek 0.75 ± 0.1 µm. Con 0.75 µm le righe calano del ~21–25 %.
4. **Si attivo:** default dell'app 500 µm (offerta ORTEC e manuale X-123); `lab_xrf.yml` ha {cfg_si:g} µm.
5. **Matrice di risposta Geant4 non inclusa nel repository:** senza `.npz` i picchi sono gaussiani puri, senza escape
   (1.74 keV sotto ogni picco), coda e shelf. Anche il file di calibrazione MCA (`.calib`) non è nel repository.
6. **Inizio griglia a {sim['E'][0]:.3f} keV** (SpekPy): mancano solo le righe sotto ~1 keV (O, C, …). Na Kα (1.04 keV)
   è al limite della griglia; Mg Kα (1.25 keV) c'è, ma in aria è fortemente assorbita.
7. **Dead time** stimato con modello paralizzabile, τ = (τ/Tpeak) × Tpeak; il peaking time di default (4.8 µs) è una
   STIMA. Il `.mca` esportato ha REAL_TIME = LIVE_TIME/(1 − DT). Pile-up e somme non modellati.
8. **Geometria:** in modalità parametrica è nel piano x–z con il rivelatore puntato sullo spot (`det_normal = −r_det`);
   in modalità config si usano i vettori di `lab_xrf.yml` così come sono.
9. **Etichette Ag K** (Rayleigh e Compton) mostrate solo sopra il bordo K dell'Ag ({core.AG_K_EDGE_keV} keV).

#### Unità
- Tabella righe: *ingresso rivelatore* = ph/s che arrivano al rivelatore (geometria + aria), *rivelati* = × T_Be × QE_Si.
- Concentrazioni: frazioni in massa normalizzate a somma 1.
- Spettro: ph/s/keV (densità) oppure conteggi per canale nel live time scelto.
""")
