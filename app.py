"""
XRF sim — interfaccia grafica Streamlit
=======================================

GUI per il simulatore a parametri fondamentali (pacchetto `xrf_model` in ./src).
Il pacchetto non viene modificato: la config è costruita in memoria partendo da
configs/lab_xrf.yml e `run_one_grid` è chiamato così com'è. Le funzioni di
calcolo stanno in xrf_gui_core.py (importabili e testabili senza Streamlit).
I testi visibili all'utente sono in i18n.py (Italiano / English).

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

from i18n import LANG_LABELS, set_lang, t

st.set_page_config(page_title="XRF sim", page_icon="🌙", layout="wide", initial_sidebar_state="expanded")


def init_language() -> None:
    if "lang" not in st.session_state:
        st.session_state["lang"] = "it"
    set_lang(st.session_state["lang"])


def language_selector(container) -> None:
    container.radio(
        t("language"),
        options=["it", "en"],
        format_func=lambda x: LANG_LABELS[x],
        horizontal=True,
        key="lang",
    )
    set_lang(st.session_state["lang"])


init_language()

try:
    import xrf_gui_core as core
except Exception as exc:   # pragma: no cover
    language_selector(st)
    st.error(t("err_import", exc=exc))
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
    language_selector(st)
    st.markdown(t("login_title"))
    with st.form("login"):
        entered = st.text_input(t("password"), type="password")
        ok = st.form_submit_button(t("btn_enter"), type="primary")
    if ok:
        if core.check_password(entered, _PW):
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error(t("err_bad_password"))
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


def atm_label(a: str) -> str:
    return t("atm_air") if a == "air" else t("atm_vacuum")


def filter_label(m: str) -> str:
    return t("filter_none") if m == "nessuno" else m


cfg0 = base_config(core.file_stamp(core.CONFIG_PATH))
status = core.local_files_status(cfg0)
gdef = core.geometry_defaults(cfg0)
geo_param_ok, geo_reason = core.geometry_is_parametric(cfg0)
samples = core.list_samples(cfg0)
cfg_filters = {str(m): float(th) for m, th in (cfg0["tube"].get("filters") or {}).items()}
cfg_si = float(cfg0.get("si_active_layer_um", 450.0))

# ────────────────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <h1>🌙 XRF sim</h1>
  <p>{t("hero_subtitle")}</p>
  <span class="badge">{t("badge_sherman")}</span><span class="badge">{t("badge_l_lines")}</span>
  <span class="badge">{t("badge_rayleigh")}</span><span class="badge">{t("badge_compton")}</span>
  <span class="badge">{t("badge_spekpy")}</span>
  <span class="badge {'ok' if status['response'] else 'warn'}">{t("badge_geant4_ok") if status['response'] else t("badge_geant4_missing")}</span>
  <span class="badge {'ok' if status['calibration'] else 'warn'}">{t("badge_mca_ok") if status['calibration'] else t("badge_mca_missing")}</span>
</div>
""", unsafe_allow_html=True)

if (status["response_declared"] and not status["response"]) or (status["calibration_declared"] and not status["calibration"]):
    st.info(t("info_missing_files"))

# ────────────────────────────────────────────────────────────────────────────
# Sidebar: lingua, parametri fisici (form) + acquisizione
# ────────────────────────────────────────────────────────────────────────────
sb = st.sidebar
language_selector(sb)
if _PW:
    if sb.button(t("btn_sign_out"), icon="🔒"):
        st.session_state["auth_ok"] = False
        st.rerun()
else:
    sb.caption(t("caption_no_password"))
sb.markdown(t("sidebar_params"))

with sb.form("phys"):
    st.markdown(t("section_sample"))
    default_sample = cfg0["samples"][0] if cfg0["samples"][0] in samples else samples[0]
    sample = st.selectbox(t("label_composition"), samples, index=samples.index(default_sample))
    custom_file = st.file_uploader(t("label_upload_yml"), type=["yml", "yaml"], key="custom_yml")
    secondary = st.toggle(t("toggle_secondary"), value=bool(cfg0.get("include_secondary", True)))

    st.markdown(t("section_tube"))
    c1, c2 = st.columns(2)
    kv = c1.slider(t("label_voltage"), 10, 50, int(cfg0["tube"]["kvp"]))
    uA = c2.slider(t("label_current"), 5, 200, 50, step=5)
    target_um = st.number_input(t("label_target"), 0.3, 3.0,
                                float(cfg0["tube"]["transmission_thickness_um"]), 0.05,
                                help=t("help_target"))
    keep_filters = st.multiselect(
        t("label_keep_filters"), list(cfg_filters), default=list(cfg_filters),
        format_func=lambda m: f"{m} {cfg_filters[m]:g} mm",
        help=t("help_keep_filters"))
    c1, c2 = st.columns(2)
    filter_mat = c1.selectbox(t("label_extra_filter"), core.FILTER_MATERIALS, index=0,
                              format_func=filter_label,
                              help=t("help_extra_filter"))
    filter_mm = c2.number_input(t("label_thickness"), 0.0, 2.0, 0.0, 0.005, format="%.3f")

    st.markdown(t("section_env"))
    atmosphere = st.radio(t("label_atmosphere"), ["air", "vacuum"], horizontal=True,
                          index=0 if cfg0.get("atmosphere", "vacuum") == "air" else 1,
                          format_func=atm_label)
    grid_n = st.select_slider(t("label_grid"), options=[1, 2, 3, 4, 5, 6, 8, 10, 15, 20],
                              value=int(cfg0["grid_n_x"]))
    grid_mode = st.radio(t("label_illuminated_area"), ["config", "collimatore", "manuale"], horizontal=True,
                         format_func=lambda m: {"config": t("area_config"), "collimatore": t("area_collimator"),
                                                "manuale": t("area_manual")}[m],
                         help=t("help_illuminated_area"))
    beam_D_mm = st.number_input(t("label_beam_d"), 0.5, 20.0, 2.0, 0.1, help=t("help_beam_d"))
    c1, c2 = st.columns(2)
    hx_man = c1.number_input(t("label_hx"), 0.01, 3.0, float(cfg0["grid_x_max"]), 0.01)
    hy_man = c2.number_input(t("label_hy"), 0.01, 3.0, float(cfg0["grid_y_max"]), 0.01)

    with st.expander(t("exp_geometry"), expanded=not geo_param_ok):
        geo_mode = st.radio(t("label_geo_mode"), ["config", "parametrica"], horizontal=True,
                            index=1 if geo_param_ok else 0,
                            format_func=lambda m: {"config": t("geo_config"), "parametrica": t("geo_parametric")}[m])
        if not geo_param_ok:
            st.warning(t("warn_geo_not_parametric", reason=geo_reason))
        c1, c2 = st.columns(2)
        theta_src = c1.number_input(t("label_theta_src"), 0.0, 85.0, round(gdef["theta_src"], 2), 0.5)
        D_src = c2.number_input(t("label_d_src"), 0.3, 30.0, round(gdef["D_src"], 3), 0.05)
        theta_det = c1.number_input(t("label_theta_det"), 0.0, 85.0, round(gdef["theta_det"], 2), 0.5)
        D_det = c2.number_input(t("label_d_det"), 0.3, 30.0, round(gdef["D_det"], 3), 0.05)
        src_vac = c1.number_input(t("label_src_vac"), 0.0, 10.0,
                                  float(cfg0.get("src_vacuum_path_cm", 0.0)), 0.05)
        det_vac = c2.number_input(t("label_det_vac"), 0.0, 10.0,
                                  float(cfg0.get("det_vacuum_path_cm", 0.0)), 0.05)
        st.caption(t("caption_geo"))

    with st.expander(t("exp_detector")):
        c1, c2 = st.columns(2)
        det_area_mm2 = c1.number_input(t("label_det_area"), 1.0, 100.0, float(cfg0["det_area"]) * 100, 1.0,
                                       help=t("help_det_area"))
        be_um = c2.number_input(t("label_be"), 0.0, 100.0, float(cfg0.get("be_window_mm", 0.0127)) * 1000, 0.1)
        si_um = c1.number_input(t("label_si"), 100.0, 1000.0, 500.0, 10.0, help=t("help_si"))
        c1.caption(t("caption_si_yml", cfg_si=cfg_si))
        fano = c2.number_input(t("label_fano"), 0.05, 0.2, float(cfg0.get("detector_fano", 0.118)), 0.001, format="%.4f")
        w_ev = c1.number_input(t("label_w"), 3.5, 3.8, float(cfg0.get("detector_w_eV", 3.65)), 0.01)
        noise_ev = c2.number_input(t("label_noise"), 0.0, 60.0, float(cfg0.get("detector_noise_eV", 12.7)), 0.1)
        use_response = st.toggle(t("toggle_geant4"), value=True, disabled=not status["response"])

    if grid_n >= 6:
        st.warning(t("warn_grid_time", n=grid_n, sec=1.0 + 0.25 * grid_n ** 2))
    submitted = st.form_submit_button(t("btn_simulate"), type="primary", width="stretch")

sb.markdown(t("sidebar_acq"))
live_time = sb.number_input(t("label_live_time"), 1.0, 36000.0, 300.0, 10.0)
poisson_on = sb.toggle(t("toggle_poisson"), value=True)
c1, c2 = sb.columns([1, 1])
seed_mode = c1.radio(t("label_seed"), ["fisso", "casuale"], horizontal=False, label_visibility="collapsed",
                     format_func=lambda m: t("seed_fisso") if m == "fisso" else t("seed_casuale"))
if "seed_fixed" not in st.session_state:
    st.session_state["seed_fixed"] = 42
if "seed_rand" not in st.session_state:
    st.session_state["seed_rand"] = pysecrets.randbelow(10**9)
if seed_mode == "fisso":
    c2.number_input(t("label_seed"), 0, 10**9, key="seed_fixed", label_visibility="collapsed")
def _new_realization(mode):
    if mode == "fisso":
        st.session_state["seed_fixed"] = int(st.session_state["seed_fixed"]) + 1
    else:
        st.session_state["seed_rand"] = pysecrets.randbelow(10**9)


sb.button(t("btn_new_realization"), width="stretch", disabled=not poisson_on,
          on_click=_new_realization, args=(seed_mode,))
seed = int(st.session_state["seed_fixed"] if seed_mode == "fisso" else st.session_state["seed_rand"])
sb.caption(t("caption_seed", seed=seed))
c1, c2 = sb.columns(2)
tpeak_us = c1.number_input(t("label_tpeak"), 0.05, 30.0, core.TPEAK_DEFAULT_us, 0.1,
                           help=t("help_tpeak"))
tau_ratio = c2.number_input(t("label_tau_ratio"), 0.5, 5.0, core.TAU_OVER_TPEAK, 0.1,
                            help=t("help_tau"))
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
        sb.error(t("err_invalid_comp", exc=exc))

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
    extra = t("caption_si_mismatch_geant4") if status["response"] else ""
    sb.caption(t("caption_si_mismatch", si=P["si_um"], cfg=cfg_si) + extra)

# ────────────────────────────────────────────────────────────────────────────
# Simulazione
# ────────────────────────────────────────────────────────────────────────────
power_W = P["kv"] * P["uA"] * 1e-3
try:
    with st.spinner(t("spinner", sample=P["sample"], kv=P["kv"], n=P["grid_n"])):
        sim1 = simulate(core.phys_key(P, cfg0), {k: P[k] for k in core.PHYS_KEYS}, cfg0)
except Exception as exc:
    st.error(t("err_sim_failed"))
    st.exception(exc)
    st.stop()
sim = core.scaled(sim1, P["uA"])
cfgP = sim["cfg"]

with sb.expander(t("exp_composition"), expanded=False):
    cdf = pd.DataFrame({t("col_el"): list(sim["comp"]), t("col_wt"): [100 * v for v in sim["comp"].values()]})
    cdf[t("col_ppm")] = cdf[t("col_wt")] * 1e4
    st.dataframe(cdf.sort_values(t("col_wt"), ascending=False), hide_index=True, width="stretch",
                 column_config={t("col_wt"): st.column_config.NumberColumn(format="%.3f"),
                                t("col_ppm"): st.column_config.NumberColumn(format="%.0f")})
    st.caption(t("caption_comp_norm"))

# ── spettro misurato ────────────────────────────────────────────────────────
meas = None
with st.expander(t("exp_measured"), expanded=False):
    up = st.file_uploader(t("label_mca_file"), type=["mca", "MCA", "txt"], key="meas")
    if up is not None:
        try:
            meas = core.parse_mca(up.getvalue())
        except core.McaError as exc:
            st.error(t("err_invalid_file", exc=exc))
        except Exception as exc:
            st.error(t("err_read_file", typ=type(exc).__name__, exc=exc))
    if meas is not None:
        n = len(meas["counts"])
        if meas["cal_note"]:
            st.warning(t("warn_cal_note", note=meas["cal_note"]))
        elif meas["calib"] is None:
            st.warning(t("warn_no_cal"))
        cal = meas["calib"] or (0.0, 0.006, 0.0)
        c1, c2, c3, c4 = st.columns(4)
        A = c1.number_input(t("label_cal_a"), -5.0, 5.0, float(cal[0]), 0.001, format="%.5f")
        B = c2.number_input(t("label_cal_b"), core.B_RANGE_keV[0], core.B_RANGE_keV[1], float(cal[1]), 0.0001, format="%.6f")
        Cq = c3.number_input(t("label_cal_c"), -1e-5, 1e-5, float(cal[2]), 1e-9, format="%.3e")
        if meas["live_missing"]:
            real_txt = t("warn_live_missing_real", real=meas["real"]) if meas["real"] else t("warn_live_missing_end")
            st.warning(t("warn_live_missing", real=real_txt))
        lt_m = c4.number_input(t("label_meas_live"), 0.001, 1e6, float(meas["live"] or meas["real"] or 1.0), 1.0)
        meas.update(A=A, B=B, Cq=Cq, live=lt_m)
        c1, c2 = st.columns(2)
        meas["align"] = c1.toggle(t("toggle_align"), value=True)
        meas["norm"] = c2.radio(t("label_norm"), ["nessuna", "live time", "su un picco"], index=1, horizontal=True,
                                format_func=lambda v: {"nessuna": t("norm_none"), "live time": t("norm_live"),
                                                       "su un picco": t("norm_peak")}[v])
        if meas["calib"]:
            cal_txt = t("cal_from_file",
                        label=meas["cal_label"] or t("cal_no_label"),
                        unit=meas["cal_unit"],
                        deg=t("cal_quadratic") if meas["cal_deg"] == 2 else t("cal_linear"))
        else:
            cal_txt = t("cal_manual")
        st.caption(t("caption_meas", n=n, live=meas["live"], real=meas["real"] or "—",
                     cal=cal_txt, tot=meas["counts"].sum()))
        info = {t("info_tpea"): meas["tpeak_us"], t("info_gain"): meas["gain"],
                t("info_fast"): meas["fast"], t("info_slow"): meas["slow"],
                t("info_meas_dt"): None if meas["meas_dt"] is None else round(100 * meas["meas_dt"], 2)}
        info = {k: v for k, v in info.items() if v is not None}
        if info or meas["dp5"] or meas["status"]:
            with st.popover(t("popover_dp5")):
                if info:
                    st.dataframe(pd.DataFrame({t("col_item"): list(info), t("col_value"): [str(v) for v in info.values()]}),
                                 hide_index=True, width="stretch")
                if meas["dp5"]:
                    st.markdown("**DP5 CONFIGURATION**")
                    st.dataframe(pd.DataFrame({t("col_key"): list(meas["dp5"]), t("col_value"): list(meas["dp5"].values())}),
                                 hide_index=True, width="stretch", height=240)
                if meas["status"]:
                    st.markdown("**DPP STATUS**")
                    st.dataframe(pd.DataFrame({t("col_item"): list(meas["status"]), t("col_value"): list(meas["status"].values())}),
                                 hide_index=True, width="stretch", height=240)

# ── canali MCA ──────────────────────────────────────────────────────────────
if meas is not None and meas.get("align"):
    calib, n_ch, ch_src = (meas["A"], meas["B"], meas["Cq"]), len(meas["counts"]), t("ch_src_measured")
elif status["calibration"]:
    calib, n_ch, ch_src = core.read_calibration(cfg0["mca_calibration"]), int(cfg0["mca_channels"]), t("ch_src_config")
else:
    step = float(np.mean(np.diff(sim["sE"])))
    calib, n_ch = (float(sim["sE"][0]), step, 0.0), len(sim["sE"])
    ch_src = t("ch_src_grid", step=step * 1000)
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
m[0].metric(t("metric_total_counts"), compact(float(counts.sum())),
            help=t("help_total_counts", lt=live_time,
                   poisson=t("with_poisson") if poisson_on else t("without_poisson")))
m[1].metric(t("metric_rate"), compact(total_rate, "cps"), help=t("help_rate"))
m[2].metric(t("metric_dead"), f"{100*dead:.1f} %",
            help=t("help_dead", tau=tau_us, ratio=tau_ratio, tpeak=tpeak_us, rt=real_time_est))
m[3].metric(t("metric_power"), f"{power_W:.2f} W", delta=t("metric_power_max", w=core.MINI_X2_MAX_W),
            delta_color="inverse" if power_W > core.MINI_X2_MAX_W else "off")
m[4].metric(t("metric_fwhm"), f"{fwhm59:.0f} eV")
m[5].metric(t("metric_compute"), f"{sim['dt']:.1f} s", help=t("help_compute"))
if power_W > core.MINI_X2_MAX_W:
    st.error(t("err_power", kv=P["kv"], uA=P["uA"], power=power_W,
               imax=core.MINI_X2_MAX_W / P["kv"] * 1000))
if sim["n_valid"] < P["grid_n"] ** 2:
    st.warning(t("warn_cells", n_valid=sim["n_valid"], n_tot=P["grid_n"] ** 2))
if meas is not None and meas.get("meas_dt") is not None:
    src = t("dt_fast_slow") if meas["fast"] else t("dt_live_real")
    st.caption(t("caption_dt_meas", meas=100 * meas["meas_dt"], src=src, est=100 * dead))

tab_spec, tab_lines, tab_cmp, tab_inst, tab_notes = st.tabs(
    [t("tab_spectrum"), t("tab_lines"), t("tab_compare"), t("tab_instrument"), t("tab_notes")])
psi = core.scattering_angle_deg(cfgP)
emax_default = float(min(P["kv"], 40.0))

# ── TAB SPETTRO ─────────────────────────────────────────────────────────────
with tab_spec:
    c = st.columns([1.2, 1.2, 2.2, 1.4])
    ymode = c[0].radio(t("label_yaxis"), ["conteggi/canale", "ph/s/keV"],
                       format_func=lambda v: t("y_counts") if v == "conteggi/canale" else t("y_density"))
    yscale = c[1].radio(t("label_scale"), ["log", "lin"],
                        format_func=lambda v: t("scale_log") if v == "log" else t("scale_lin"))
    erange = c[2].slider(t("label_energy"), 0.0, float(P["kv"]), (1.0, emax_default), 0.1)
    show_comp = c[3].multiselect(t("label_components"), ["fluorescenza", "coerente", "incoerente"],
                                 default=[] if sim["used_response"] else ["coerente", "incoerente"],
                                 disabled=sim["used_response"],
                                 format_func=lambda n: t({"fluorescenza": "comp_fluo", "coerente": "comp_coh",
                                                          "incoerente": "comp_inc"}[n]),
                                 help=t("help_components"))
    c = st.columns([1, 1, 1, 3])
    show_labels = c[0].toggle(t("toggle_labels"), value=True)
    show_ar = c[1].toggle(t("toggle_ar"), value=True)
    show_expected = c[2].toggle(t("toggle_expected"), value=poisson_on)
    n_labels = c[3].slider(t("label_n_labels"), 3, 40, 16)
    to_counts = ymode == "conteggi/canale"

    def as_y(ch_counts):          # conteggi per canale → asse scelto
        return ch_counts if to_counts else ch_counts / (live_time * ch_w)

    has_meas = meas is not None
    fig = make_subplots(rows=2 if has_meas else 1, cols=1, shared_xaxes=True,
                        row_heights=[0.76, 0.24] if has_meas else [1.0], vertical_spacing=0.04)
    if poisson_on:
        fig.add_trace(go.Scatter(x=ch_E, y=as_y(counts), mode="lines", line=dict(color=C["poisson"], width=1, shape="hvh"),
                                 name=t("trace_sim_poisson", seed=seed)), 1, 1)
    if show_expected or not poisson_on:
        fig.add_trace(go.Scatter(x=ch_E, y=as_y(expected), mode="lines", line=dict(color=C["total"], width=2),
                                 name=t("trace_sim_expected") + (t("trace_geant4_suffix") if sim["used_response"] else "")), 1, 1)
    comp_map = {"fluorescenza": ("comp_fluo", C["fluo"], "comp_fluo"),
                "coerente": ("comp_coh", C["coh"], "comp_coh"),
                "incoerente": ("comp_inc", C["inc"], "comp_inc")}
    for name in show_comp:
        key, col, tkey = comp_map[name]
        fig.add_trace(go.Scatter(x=ch_E, y=as_y(core.component_on_channels(sim, key, edges) * live_time),
                                 mode="lines", line=dict(color=col, width=1.3, dash="dot"), name=t(tkey)), 1, 1)

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
                choice = st.selectbox(t("label_norm_peak"), opts)
                E0 = float(choice.split("(")[1].split()[0])
                fw = 2.3548 * float(core.sigma_kev(E0, P["fano"], P["noise_ev"], P["w_ev"]))
                win = np.array([E0 - 0.9 * fw, E0 + 0.9 * fw])
                sim_w = float(core.rate_on_channels(sim, win)[0] * live_time)
                sel = (m_E >= win[0]) & (m_E <= win[1])
                meas_w = float(meas["counts"][sel].sum())
                k_meas = sim_w / meas_w if meas_w > 0 else 1.0
        y_m = meas["counts"] * k_meas if to_counts else meas["counts"] * k_meas / (live_time * m_w)
        fig.add_trace(go.Scatter(x=m_E, y=y_m, mode="lines", line=dict(color=C["meas"], width=1.2, shape="hvh"),
                                 name=t("trace_measured", k=k_meas)), 1, 1)
        if meas.get("align"):
            res = core.residuals(meas["counts"], expected, k_meas)
            fig.add_trace(go.Scatter(x=ch_E, y=res, mode="lines", line=dict(color=C["meas"], width=1),
                                     name=t("trace_residuals"), showlegend=False), 2, 1)
            fig.update_yaxes(title_text=t("axis_residuals"), row=2, col=1, gridcolor=C["grid"], zeroline=True,
                             zerolinecolor="rgba(229,233,242,0.4)")
        else:
            fig.add_annotation(text=t("anno_residuals"), row=2, col=1,
                               x=0.5, y=0.5, xref="x domain", yref="y2 domain", showarrow=False,
                               font=dict(color="#8C96B0"))

    if show_ar:
        fig.add_vrect(x0=core.AR_ZONE[0], x1=core.AR_ZONE[1], fillcolor="rgba(248,113,113,0.10)", line_width=0,
                      annotation_text=t("anno_ar"), annotation_position="top left",
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
                     title_text=t("axis_counts") if to_counts else t("axis_density"), gridcolor=C["grid"], row=1, col=1)
    if yscale == "log":
        sel = (ref_x >= erange[0]) & (ref_x <= erange[1])
        ymax = float(np.max(ref_y[sel])) if sel.any() else 1.0
        floor = 0.5 if to_counts else ymax * 1e-6
        fig.update_yaxes(range=[math.log10(max(floor, 1e-12)), math.log10(max(ymax, 1e-9) * 6)], row=1, col=1)
    fig.update_xaxes(range=list(erange), gridcolor=C["grid"], row=1, col=1)
    fig.update_xaxes(title_text=t("axis_energy"), gridcolor=C["grid"], row=2 if has_meas else 1, col=1)
    fig.update_layout(**PLOT_BG, height=620 if has_meas else 560, margin=dict(l=60, r=20, t=40, b=50),
                      hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
                      title=dict(text=t("plot_title", sample=P["sample"], kv=P["kv"], uA=P["uA"],
                                        atm=atm_label(P["atmosphere"]), lt=live_time, n=P["grid_n"]),
                                 font=dict(size=14), x=0.0, y=0.995))
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "toImageButtonOptions":
                    {"format": "png", "filename": f"xrf_{P['sample']}_{P['kv']}kV", "scale": 2}})
    st.markdown(f"<span class='small-note'>{t('caption_channels', src=ch_src, n=n_ch, psi=psi, emin=sim['E'][0])}</span>",
                unsafe_allow_html=True)

    st.markdown(t("section_export"))
    c = st.columns([2, 1, 1, 1, 1, 1])
    filt_txt = ", ".join(f"{m_} {th:g} mm" for m_, th in cfgP["tube"]["filters"].items())
    default_label = (f"{P['sample']} · {P['kv']} kV {P['uA']} µA · {atm_label(P['atmosphere'])}"
                     + (f" · {P['filter_mat']} {P['filter_mm']} mm" if P['filter_mat'] != 'nessuno' and P['filter_mm'] > 0 else ""))
    run_label = c[0].text_input(t("label_run"), default_label, label_visibility="collapsed")
    if c[1].button(t("btn_add_compare"), width="stretch"):
        runs = st.session_state.setdefault("runs", [])
        gl = core.grouped_lines(sim["fluo"])
        runs.append(dict(label=run_label, E=sim["sE"], spec=sim["spec"], params=dict(P),
                         lines={f"{r.Element} {r.Group}": r.cps for r in gl.itertuples()},
                         lines_E={f"{r.Element} {r.Group}": r.E for r in gl.itertuples()}))
        st.session_state["runs"] = runs[-8:]
        st.toast(t("toast_run_saved", label=run_label), icon="📌")

    base = f"xrf_{P['sample']}_{P['kv']}kV_{P['uA']}uA_{P['atmosphere']}"
    noise_txt = t("poisson_seed", seed=seed) if poisson_on else t("expected_word")
    desc = (f"XRF sim {P['sample']} {P['kv']}kV {P['uA']}uA {P['atmosphere']} "
            f"{t('tag_filters')}[{filt_txt}] {noise_txt} {t('dt_est_tag')} {100*dead:.1f}%")
    c[2].download_button("⬇ .mca", core.write_mca(counts, calib, live_time, real_time_est, desc),
                         file_name=base + ".mca", width="stretch",
                         help=t("help_mca", rt=real_time_est))
    spec_df = pd.DataFrame({"Energy_keV": sim["E"], "fluorescence_ph_s_keV": sim["comp_fluo"],
                            "coherent_ph_s_keV": sim["comp_coh"], "incoherent_ph_s_keV": sim["comp_inc"]})
    spec_df["total_gaussian_ph_s_keV"] = spec_df.iloc[:, 1:].sum(axis=1)
    chan_df = pd.DataFrame({"channel": np.arange(n_ch), "E_keV": ch_E, "expected_counts": expected, "counts": counts})
    csv_buf = io.StringIO()
    csv_buf.write(t("csv_comment_channels", desc=desc, lt=live_time, src=ch_src))
    chan_df.to_csv(csv_buf, index=False)
    csv_buf.write(t("csv_comment_grid"))
    spec_df.to_csv(csv_buf, index=False)
    c[3].download_button("⬇ .csv", csv_buf.getvalue().encode(), file_name=base + ".csv", width="stretch")
    lld_def_dl = st.session_state.get("lld_def", "3√B")
    xbuf = io.BytesIO()
    lines_out = sim["fluo"].copy()
    lines_out["counts_in_live_time"] = lines_out["cps_det"] * live_time
    with pd.ExcelWriter(xbuf) as xw:
        lines_out.to_excel(xw, sheet_name=t("sheet_lines"), index=False)
        core.lld_table(sim, live_time, float(sim["E"][0]), float(P["kv"]), lld_def_dl).to_excel(
            xw, sheet_name=t("sheet_lld"), index=False)
        pd.DataFrame({t("col_param"): list(P), t("col_param_value"): [str(v) for v in P.values()]}).to_excel(
            xw, sheet_name=t("sheet_params"), index=False)
    c[4].download_button("⬇ .xlsx", xbuf.getvalue(), file_name=base + t("xlsx_suffix"), width="stretch")
    cfg_text, cfg_extra = core.export_config(P, cfg0, P["uA"], live_time)
    if cfg_extra:
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(base + ".yml", cfg_text)
            for name_, txt in cfg_extra.items():
                z.writestr(name_, txt)
        c[5].download_button("⬇ config", zbuf.getvalue(), file_name=base + "_config.zip", width="stretch",
                             help=t("help_config_zip"))
    else:
        c[5].download_button("⬇ config", cfg_text.encode(), file_name=base + ".yml", width="stretch")

# ── TAB RIGHE ───────────────────────────────────────────────────────────────
with tab_lines:
    if sim["used_response"]:
        st.warning(t("warn_geant4_lines"))
    st.markdown(t("section_fluo_lines"))
    df = sim["fluo"][sim["fluo"]["Intensity_ph_s"] > 0]
    col_counts = t("col_counts_in_lt", lt=live_time)
    tab = pd.DataFrame({
        t("col_element"): df["Element"], t("col_line"): df["Line"], t("col_e_kev"): df["E_line_keV"],
        t("col_conc_ppm"): df["Concentration"] * 1e6,
        t("col_entrance"): df["Intensity_ph_s"], t("col_tbe_qe"): df["T_Be_QE"],
        t("col_detected"): df["cps_det"], col_counts: df["cps_det"] * live_time})
    min_cps = st.slider(t("label_min_cps"), 0.0, 1.0, 0.001, 0.001, format="%.3f")
    tab = tab[tab[t("col_detected")] >= min_cps].sort_values(t("col_detected"), ascending=False)
    st.dataframe(tab, hide_index=True, width="stretch", height=360, column_config={
        t("col_e_kev"): st.column_config.NumberColumn(format="%.4f"),
        t("col_conc_ppm"): st.column_config.NumberColumn(format="%.1f"),
        t("col_entrance"): st.column_config.NumberColumn(format="%.4g"),
        t("col_tbe_qe"): st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.3f"),
        t("col_detected"): st.column_config.NumberColumn(format="%.4g"),
        col_counts: st.column_config.NumberColumn(format="%.4g")})

    c1, c2 = st.columns([3, 2])
    c1.markdown(t("section_pb_lld"), unsafe_allow_html=True)
    lld_def = c2.radio(t("label_lld_def"), ["3√B", "Currie"], horizontal=True, key="lld_def",
                       help=t("help_lld_def"))
    lt = core.lld_table(sim, live_time, float(sim["E"][0]), float(P["kv"]), lld_def)
    lcol = t("col_lld_value", definition=lld_def)
    if not lt.empty:
        c1, c2 = st.columns([3, 2])
        fmt = {k: st.column_config.NumberColumn(format=f) for k, f in [
            (t("col_conc_ppm"), "%.1f"), (t("col_fwhm_ev"), "%.0f"), (t("col_n_peak"), "%.3g"),
            (t("col_b_bkg"), "%.3g"), (t("col_pb"), "%.3g"), (t("col_stat_sigma"), "%.2f"),
            (t("col_lld_ka"), "%.3g"), (t("col_lld_la"), "%.3g"), (lcol, "%.3g")]}
        c1.dataframe(lt, hide_index=True, width="stretch", height=420, column_config=fmt)
        lf = go.Figure(go.Bar(x=lt[t("col_element")] + " " + lt[t("col_line_used")], y=lt[lcol],
                              marker=dict(color=lt[t("col_e_kev")], colorscale="Tealrose", showscale=False),
                              hovertemplate="%{x}: %{y:.3g} ppm<extra></extra>"))
        lf.update_layout(**PLOT_BG, height=420, yaxis=dict(type="log", title=t("axis_lld"), gridcolor=C["grid"]),
                         margin=dict(l=50, r=10, t=30, b=40),
                         title=dict(text=t("title_lld", definition=lld_def, lt=live_time), font=dict(size=13)))
        c2.plotly_chart(lf, width="stretch", config={"displaylogo": False})
    st.caption(t("caption_lld", win=core.WIN_FWHM))

# ── TAB CONFRONTO ───────────────────────────────────────────────────────────
with tab_cmp:
    runs = st.session_state.get("runs", [])
    if not runs:
        st.info(t("info_no_runs"))
    else:
        c1, c2, c3 = st.columns([1, 1, 3])
        cmp_scale = c1.radio(t("label_scale"), ["log", "lin"], horizontal=True, key="cmp_scale",
                             format_func=lambda v: t("scale_log") if v == "log" else t("scale_lin"))
        cmp_norm = c2.toggle(t("toggle_norm_ua"), value=False)
        if c3.button(t("btn_clear_compare")):
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
                           xaxis=dict(title=t("axis_energy"), range=[1, 26], gridcolor=C["grid"]),
                           yaxis=dict(title=t("axis_density_per_ua") if cmp_norm else "ph/s/keV",
                                      type="log" if cmp_scale == "log" else "linear", gridcolor=C["grid"]))
        st.plotly_chart(fig2, width="stretch", config={"displaylogo": False})
        keys = sorted({k for r in runs for k in r["lines"]}, key=lambda k: -max(r["lines"].get(k, 0) for r in runs))[:20]
        keys = sorted(keys, key=lambda k: next(r["lines_E"][k] for r in runs if k in r["lines_E"]))
        rows = []
        for k in keys:
            row = {t("col_line"): k, t("col_e_kev"): next(r["lines_E"][k] for r in runs if k in r["lines_E"])}
            ref_v = runs[0]["lines"].get(k, 0.0) / (runs[0]["params"]["uA"] if cmp_norm else 1.0)
            for i, r in enumerate(runs):
                v = r["lines"].get(k, 0.0) / (r["params"]["uA"] if cmp_norm else 1.0)
                row[f"[{i+1}] cps"] = v
                if i > 0:
                    row[f"[{i+1}]/[1]"] = v / ref_v if ref_v > 0 else np.nan
            rows.append(row)
        st.markdown(t("section_main_lines") + " · ".join(f"[{i+1}] {r['label']}" for i, r in enumerate(runs)))
        rdf = pd.DataFrame(rows)
        st.dataframe(rdf, hide_index=True, width="stretch",
                     column_config={c_: st.column_config.NumberColumn(format="%.4g")
                                    for c_ in rdf.columns if c_ != t("col_line")})

# ── TAB STRUMENTO ───────────────────────────────────────────────────────────
with tab_inst:
    c1, c2 = st.columns(2)
    sx, sz, dx, dz = cfgP["src_x"], cfgP["src_z"], cfgP["det_x"], cfgP["det_z"]
    g = go.Figure()
    g.add_shape(type="rect", x0=-6, x1=6, y0=-0.6, y1=0, fillcolor="#3b3f4f", line_width=0)
    g.add_shape(type="rect", x0=-P["grid_hx"], x1=P["grid_hx"], y0=-0.06, y1=0.06,
                fillcolor="rgba(45,212,191,0.8)", line_width=0)
    for (x, z, col, name) in [(sx, sz, "#FBBF24", t("name_tube")), (dx, dz, "#F472B6", t("name_det"))]:
        g.add_trace(go.Scatter(x=[x, 0], y=[z, 0], mode="lines", line=dict(color=col, width=2, dash="dot"),
                               showlegend=False, hoverinfo="skip"))
        g.add_trace(go.Scatter(x=[x], y=[z], mode="markers+text", text=[name], textposition="top center",
                               marker=dict(size=16, color=col, symbol="square"), name=name))
    g.add_annotation(x=0, y=-0.35, text=t("anno_illuminated", hx=P["grid_hx"], hy=P["grid_hy"]),
                     showarrow=False, font=dict(color="#CFD6E6", size=11))
    off_plane = abs(cfgP["src_y"]) > 1e-6 or abs(cfgP["det_y"]) > 1e-6
    g.update_layout(**PLOT_BG, height=360, margin=dict(l=40, r=10, t=40, b=40), showlegend=False,
                    title=dict(text=t("title_geo",
                                      yoff=t("geo_y_off") if off_plane else "",
                                      psi=psi,
                                      mode=t("geo_mode_config") if P["geo_mode"] == "config" else t("geo_mode_param")),
                               font=dict(size=13)),
                    xaxis=dict(title=t("axis_x_cm"), range=[-7, 7], gridcolor=C["grid"]),
                    yaxis=dict(title=t("axis_z_cm"), range=[-0.8, max(sz, dz) * 1.35], gridcolor=C["grid"], scaleanchor="x", scaleratio=1))
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
    for y, name, col, dash in [(TBe, t("eff_be", um=P["be_um"]), C["coh"], "dot"),
                               (QE, t("eff_si", um=P["si_um"]), C["inc"], "dot"),
                               (Tair, t("eff_atm_path", atm=atm_label(P["atmosphere"]), L=L_out), C["fluo"], "dot"),
                               (TBe * QE * Tair, t("eff_total"), C["total"], "solid")]:
        e.add_trace(go.Scatter(x=Eg, y=y, name=name, line=dict(color=col, dash=dash, width=2 if dash == "solid" else 1.4)))
    e.update_layout(**PLOT_BG, height=360, margin=dict(l=50, r=10, t=40, b=40), hovermode="x unified",
                    title=dict(text=t("title_efficiency"), font=dict(size=13)),
                    legend=dict(orientation="h", y=-0.25), xaxis=dict(title=t("axis_energy"), type="log", gridcolor=C["grid"]),
                    yaxis=dict(title=t("axis_transmission"), range=[0, 1.02], gridcolor=C["grid"]))
    c2.plotly_chart(e, width="stretch", config={"displaylogo": False})

    c1, c2 = st.columns(2)
    try:
        tcfg = dict(cfgP["tube"])
        tcfg["current_mA"] = P["uA"] / 1000.0
        with contextlib.redirect_stdout(io.StringIO()):
            Et, Pt = core.tube_source_term(tcfg, Path(cfg0["_repo_root"]) / ".cache" / "tube")
        tu = go.Figure()
        tu.add_trace(go.Scatter(x=Et, y=Pt, name=t("trace_tube_out"), line=dict(color=C["inc"], width=1.6)))
        if P["atmosphere"] == "air":
            tu.add_trace(go.Scatter(x=Et, y=Pt * np.exp(-mu_air(Et) * L_in), name=t("trace_after_air", L=L_in),
                                    line=dict(color=C["total"], width=1.6)))
        tu.update_layout(**PLOT_BG, height=360, margin=dict(l=60, r=10, t=40, b=40), hovermode="x unified",
                         title=dict(text=t("title_tube", kv=P["kv"], um=P["target_um"],
                                           filt=filt_txt or t("filter_none")), font=dict(size=13)),
                         legend=dict(orientation="h", y=-0.25), xaxis=dict(title=t("axis_energy"), gridcolor=C["grid"]),
                         yaxis=dict(title=t("axis_tube_y"), type="log", gridcolor=C["grid"],
                                    range=[math.log10(Pt.max()) - 5, math.log10(Pt.max()) + 0.3]))
        c1.plotly_chart(tu, width="stretch", config={"displaylogo": False})
    except Exception as exc:
        c1.warning(t("warn_tube", exc=exc))

    cdf = pd.DataFrame({"El": list(sim["comp"]), "w": list(sim["comp"].values())})
    b = go.Figure(go.Bar(x=cdf["El"], y=cdf["w"] * 1e6, marker=dict(color=np.log10(cdf["w"].clip(lower=1e-12)), colorscale="Tealrose"),
                         hovertemplate="%{x}: %{y:.4g} ppm<extra></extra>"))
    b.update_layout(**PLOT_BG, height=360, margin=dict(l=60, r=10, t=40, b=40),
                    title=dict(text=t("title_comp", sample=P["sample"]), font=dict(size=13)),
                    yaxis=dict(title=t("axis_ppm_mass"), type="log", gridcolor=C["grid"]))
    c2.plotly_chart(b, width="stretch", config={"displaylogo": False})

    with st.expander(t("exp_config_log")):
        st.code(cfg_text, language="yaml")
        st.code(sim["log"] or t("log_cache"), language="text")

# ── TAB NOTE ────────────────────────────────────────────────────────────────
with tab_notes:
    st.markdown(t("notes_md", cfg_si=cfg_si, e_min=sim["E"][0], ag_edge=core.AG_K_EDGE_keV))
