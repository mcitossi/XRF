"""Traduzioni IT/EN dell'interfaccia XRF sim (nessuna dipendenza da Streamlit)."""
from __future__ import annotations

import string
from typing import Any

LANGS = ("it", "en")
DEFAULT_LANG = "it"
LANG_LABELS = {"it": "Italiano", "en": "English"}

_lang = DEFAULT_LANG

_formatter = string.Formatter()


def set_lang(lang: str) -> str:
    global _lang
    _lang = lang if lang in LANGS else DEFAULT_LANG
    return _lang


def get_lang() -> str:
    return _lang


def t(key: str, *, lang: str | None = None, **kwargs: Any) -> str:
    """Restituisce il testo per `key` nella lingua corrente (o in `lang`)."""
    if key not in STRINGS:
        raise KeyError(f"i18n: missing key {key!r}")
    loc = lang if lang in LANGS else _lang
    entry = STRINGS[key]
    if loc not in entry or entry[loc] is None or entry[loc] == "":
        raise KeyError(f"i18n: missing {loc} text for key {key!r}")
    text = entry[loc]
    return text.format(**kwargs) if kwargs else text


def placeholders(text: str) -> set[str]:
    names = set()
    for _, field, _, _ in _formatter.parse(text):
        if field:
            names.add(field.split("!")[0].split(":")[0])
    return names


def translation_problems() -> list[str]:
    """Controlli strutturali: entrambe le lingue, testi non vuoti, stessi placeholder."""
    problems = []
    for key, entry in STRINGS.items():
        if not isinstance(entry, dict):
            problems.append(f"{key}: value is not a dict")
            continue
        for loc in LANGS:
            if loc not in entry:
                problems.append(f"{key}: missing '{loc}'")
            elif not isinstance(entry[loc], str) or entry[loc] == "":
                problems.append(f"{key}: empty or non-string '{loc}'")
        if all(isinstance(entry.get(loc), str) and entry[loc] for loc in LANGS):
            pit, pen = placeholders(entry["it"]), placeholders(entry["en"])
            if pit != pen:
                problems.append(f"{key}: placeholders differ it={sorted(pit)} en={sorted(pen)}")
    return problems


# ────────────────────────────────────────────────────────────────────────────
# Testi. Chiavi stabili; i valori interni dei widget restano in app.py.
# ────────────────────────────────────────────────────────────────────────────
STRINGS: dict[str, dict[str, str]] = {
    # lingua
    "language": {
        "it": "Lingua",
        "en": "Language",
    },

    # accesso
    "login_title": {
        "it": "## 🔒 XRF sim",
        "en": "## 🔒 XRF sim",
    },
    "password": {
        "it": "Password",
        "en": "Password",
    },
    "btn_enter": {
        "it": "Entra",
        "en": "Sign in",
    },
    "err_bad_password": {
        "it": "Password errata.",
        "en": "Incorrect password.",
    },
    "btn_sign_out": {
        "it": "Esci",
        "en": "Sign out",
    },
    "caption_no_password": {
        "it": "Accesso senza password (uso locale). Per proteggerla: `app_password` in "
             "`.streamlit/secrets.toml` o variabile `XRF_APP_PASSWORD`.",
        "en": "No password (local use). To protect the app: set `app_password` in "
             "`.streamlit/secrets.toml` or the `XRF_APP_PASSWORD` environment variable.",
    },
    "err_import": {
        "it": "Non trovo `xrf_gui_core.py` o il pacchetto `xrf_model`. Metti `app.py` e "
             "`xrf_gui_core.py` nella cartella del repo (accanto a `src/` e `configs/`) "
             "e attiva il venv del repo.\n\nErrore: {exc}",
        "en": "Cannot find `xrf_gui_core.py` or the `xrf_model` package. Place `app.py` and "
             "`xrf_gui_core.py` in the repo folder (next to `src/` and `configs/`) "
             "and activate the repo venv.\n\nError: {exc}",
    },

    # intestazione
    "hero_subtitle": {
        "it": "Interfaccia per il codice a parametri fondamentali <code>xrf_model</code> (v3) — "
             "Mini-X2 Ag + X-123 FAST-SDD, INAF Trieste",
        "en": "Interface to the fundamental-parameter code <code>xrf_model</code> (v3) — "
             "Mini-X2 Ag + X-123 FAST-SDD, INAF Trieste",
    },
    "badge_sherman": {
        "it": "Sherman + secondaria K",
        "en": "Sherman + K secondary",
    },
    "badge_l_lines": {
        "it": "righe L + Coster–Kronig",
        "en": "L lines + Coster–Kronig",
    },
    "badge_rayleigh": {
        "it": "Rayleigh f0+f′",
        "en": "Rayleigh f0+f′",
    },
    "badge_compton": {
        "it": "Compton KN×S + Doppler",
        "en": "Compton KN×S + Doppler",
    },
    "badge_spekpy": {
        "it": "tubo SpekPy",
        "en": "SpekPy tube",
    },
    "badge_geant4_ok": {
        "it": "matrice Geant4: presente",
        "en": "Geant4 matrix: present",
    },
    "badge_geant4_missing": {
        "it": "matrice Geant4: assente → picchi gaussiani",
        "en": "Geant4 matrix: missing → Gaussian peaks",
    },
    "badge_mca_ok": {
        "it": "calibrazione MCA: presente",
        "en": "MCA calibration: present",
    },
    "badge_mca_missing": {
        "it": "calibrazione MCA: assente",
        "en": "MCA calibration: missing",
    },
    "info_missing_files": {
        "it": "ℹ️ `lab_xrf.yml` dichiara la matrice di risposta Geant4 (`detector_response`) e/o la "
             "calibrazione MCA (`mca_calibration`, `mca_channels`), ma i file non sono su questo computer. "
             "Sono stati tolti dalla config in memoria: i picchi sono gaussiani (Fano + rumore) e i canali sono "
             "quelli della griglia di simulazione (30 eV), oppure quelli dello spettro misurato caricato. "
             "Il file `lab_xrf.yml` non viene toccato.",
        "en": "ℹ️ `lab_xrf.yml` declares the Geant4 response matrix (`detector_response`) and/or the "
             "MCA calibration (`mca_calibration`, `mca_channels`), but the files are not on this computer. "
             "They have been dropped from the in-memory config: peaks are Gaussians (Fano + noise) and the channels are "
             "those of the simulation grid (30 eV), or those of the uploaded measured spectrum. "
             "The `lab_xrf.yml` file is left untouched.",
    },

    # sidebar — parametri
    "sidebar_params": {
        "it": "### ⚙️ Parametri",
        "en": "### ⚙️ Parameters",
    },
    "section_sample": {
        "it": "**Campione**",
        "en": "**Sample**",
    },
    "label_composition": {
        "it": "Composizione (data/compositions)",
        "en": "Composition (data/compositions)",
    },
    "label_upload_yml": {
        "it": "…oppure carica un .yml (`composition:` → frazioni o wt%)",
        "en": "…or upload a .yml (`composition:` → fractions or wt%)",
    },
    "toggle_secondary": {
        "it": "Fluorescenza secondaria K",
        "en": "K secondary fluorescence",
    },
    "section_tube": {
        "it": "**Tubo Mini-X2**",
        "en": "**Mini-X2 tube**",
    },
    "label_voltage": {
        "it": "Tensione [kV]",
        "en": "Voltage [kV]",
    },
    "label_current": {
        "it": "Corrente [µA]",
        "en": "Tube current [µA]",
    },
    "label_target": {
        "it": "Spessore target Ag [µm]",
        "en": "Ag target thickness [µm]",
    },
    "help_target": {
        "it": "lab_xrf.yml: 1.0 µm. Scheda/manuale Amptek Mini-X2: 0.75 ± 0.1 µm. "
             "Con 0.75 µm le righe calano del ~21–25 %.",
        "en": "lab_xrf.yml: 1.0 µm. Amptek Mini-X2 data sheet/manual: 0.75 ± 0.1 µm. "
             "With 0.75 µm the lines drop by ~21–25 %.",
    },
    "label_keep_filters": {
        "it": "Filtri del config (tube.filters) da mantenere",
        "en": "Config filters (tube.filters) to keep",
    },
    "help_keep_filters": {
        "it": "Il Be è la finestra d'uscita del tubo: toglierlo non è fisico, ma resta possibile.",
        "en": "Be is the tube exit window: removing it is unphysical, but it remains possible.",
    },
    "label_extra_filter": {
        "it": "Filtro aggiuntivo",
        "en": "Additional filter",
    },
    "help_extra_filter": {
        "it": "Aggiunto ai filtri del config (stesso materiale: spessori sommati).",
        "en": "Added to the config filters (same material: thicknesses are summed).",
    },
    "label_thickness": {
        "it": "Spessore [mm]",
        "en": "Thickness [mm]",
    },
    "filter_none": {
        "it": "nessuno",
        "en": "none",
    },
    "section_env": {
        "it": "**Ambiente e griglia**",
        "en": "**Environment and grid**",
    },
    "label_atmosphere": {
        "it": "Atmosfera",
        "en": "Atmosphere",
    },
    "atm_air": {
        "it": "aria",
        "en": "air",
    },
    "atm_vacuum": {
        "it": "vuoto",
        "en": "vacuum",
    },
    "label_grid": {
        "it": "Griglia n × n",
        "en": "Grid n × n",
    },
    "label_illuminated_area": {
        "it": "Area illuminata",
        "en": "Illuminated area",
    },
    "area_config": {
        "it": "da config",
        "en": "from config",
    },
    "area_collimator": {
        "it": "da collimatore",
        "en": "from collimator",
    },
    "area_manual": {
        "it": "manuale",
        "en": "manual",
    },
    "help_illuminated_area": {
        "it": "Il codice non ha un modello di footprint: la griglia È l'area illuminata.",
        "en": "The code has no footprint model: the grid IS the illuminated area.",
    },
    "label_beam_d": {
        "it": "Diametro fascio sul campione D [mm] (modo collimatore)",
        "en": "Beam diameter on the sample D [mm] (collimator mode)",
    },
    "help_beam_d": {
        "it": "Da lab_xrf.yml: per un foro di diametro d e lunghezza L con uscita a distanza g "
             "dallo spot, D sta fra d e d·(1 + g/L). 2 mm = limite inferiore (foro del collimatore).",
        "en": "From lab_xrf.yml: for an aperture of diameter d and length L with the exit at distance g "
             "from the spot, D lies between d and d·(1 + g/L). 2 mm = lower limit (collimator aperture).",
    },
    "label_hx": {
        "it": "½ lato x [cm] (manuale)",
        "en": "½ extent x [cm] (manual)",
    },
    "label_hy": {
        "it": "½ lato y [cm] (manuale)",
        "en": "½ extent y [cm] (manual)",
    },

    # geometria
    "exp_geometry": {
        "it": "📐 Geometria",
        "en": "📐 Geometry",
    },
    "label_geo_mode": {
        "it": "Modalità",
        "en": "Mode",
    },
    "geo_config": {
        "it": "usa geometria del config",
        "en": "use config geometry",
    },
    "geo_parametric": {
        "it": "parametrica θ/D",
        "en": "parametric θ/D",
    },
    "warn_geo_not_parametric": {
        "it": "La geometria di lab_xrf.yml non è descrivibile con θ/D ({reason}). "
             "In modalità parametrica verrebbe sostituita.",
        "en": "The geometry in lab_xrf.yml cannot be described with θ/D ({reason}). "
             "In parametric mode it would be replaced.",
    },
    "label_theta_src": {
        "it": "θ tubo dalla normale [°]",
        "en": "Tube θ from the normal [°]",
    },
    "label_d_src": {
        "it": "D tubo–spot [cm]",
        "en": "Tube–spot distance D [cm]",
    },
    "label_theta_det": {
        "it": "θ rivelatore [°]",
        "en": "Detector take-off angle θ [°]",
    },
    "label_d_det": {
        "it": "D spot–cristallo [cm]",
        "en": "Spot–crystal distance D [cm]",
    },
    "label_src_vac": {
        "it": "Tratto in vuoto lato tubo [cm]",
        "en": "Vacuum path on the tube side [cm]",
    },
    "label_det_vac": {
        "it": "Tratto in vuoto lato rivelatore [cm]",
        "en": "Vacuum path on the detector side [cm]",
    },
    "caption_geo": {
        "it": "Parametrica: x = ∓D·sinθ, z = D·cosθ (formule di lab_xrf.yml), rivelatore puntato sullo spot. "
             "θ è l'angolo di incidenza / take-off dalla normale alla superficie. θ/D si usano solo in modalità parametrica.",
        "en": "Parametric: x = ∓D·sinθ, z = D·cosθ (formulae from lab_xrf.yml), detector aimed at the spot. "
             "θ is the incidence / take-off angle from the surface normal. θ/D are used only in parametric mode.",
    },
    "geo_reason_off_plane": {
        "it": "tubo o rivelatore fuori dal piano x–z",
        "en": "tube or detector off the x–z plane",
    },
    "geo_reason_sides": {
        "it": "tubo e rivelatore non su lati opposti in x",
        "en": "tube and detector not on opposite sides in x",
    },
    "geo_reason_aim": {
        "it": "rivelatore non puntato sullo spot (cos = {aim:.4f})",
        "en": "detector not aimed at the spot (cos = {aim:.4f})",
    },

    # rivelatore
    "exp_detector": {
        "it": "🔬 Rivelatore X-123 FAST-SDD",
        "en": "🔬 X-123 FAST-SDD detector",
    },
    "label_det_area": {
        "it": "Area attiva [mm²]",
        "en": "Active area [mm²]",
    },
    "help_det_area": {
        "it": "25 mm² nominali, 17 mm² collimati (config).",
        "en": "25 mm² nominal, 17 mm² collimated (config).",
    },
    "label_be": {
        "it": "Finestra Be [µm]",
        "en": "Be window [µm]",
    },
    "label_si": {
        "it": "Si attivo [µm]",
        "en": "Active Si [µm]",
    },
    "help_si": {
        "it": "Default 500 µm (offerta ORTEC e manuale X-123).",
        "en": "Default 500 µm (ORTEC quotation and X-123 manual).",
    },
    "caption_si_yml": {
        "it": "lab_xrf.yml: {cfg_si:g} µm",
        "en": "lab_xrf.yml: {cfg_si:g} µm",
    },
    "label_fano": {
        "it": "Fano F",
        "en": "Fano F",
    },
    "label_w": {
        "it": "w [eV]",
        "en": "w [eV]",
    },
    "label_noise": {
        "it": "Rumore σ_el [eV rms]",
        "en": "Electronic noise σ_el [eV rms]",
    },
    "toggle_geant4": {
        "it": "Usa matrice Geant4 se presente",
        "en": "Use Geant4 matrix if available",
    },
    "warn_grid_time": {
        "it": "Griglia {n}×{n}: circa {sec:.0f} s per run.",
        "en": "Grid {n}×{n}: about {sec:.0f} s per run.",
    },
    "btn_simulate": {
        "it": "▶ Simula",
        "en": "▶ Simulate",
    },

    # acquisizione
    "sidebar_acq": {
        "it": "### ⏱️ Acquisizione",
        "en": "### ⏱️ Acquisition",
    },
    "label_live_time": {
        "it": "Live time [s]",
        "en": "Live time [s]",
    },
    "toggle_poisson": {
        "it": "Rumore Poisson",
        "en": "Poisson noise",
    },
    "label_seed": {
        "it": "Seed",
        "en": "Seed",
    },
    "seed_fisso": {
        "it": "fisso",
        "en": "fixed",
    },
    "seed_casuale": {
        "it": "casuale",
        "en": "random",
    },
    "btn_new_realization": {
        "it": "🎲 Nuova realizzazione",
        "en": "🎲 New realization",
    },
    "caption_seed": {
        "it": "Seed in uso: {seed}",
        "en": "Seed in use: {seed}",
    },
    "label_tpeak": {
        "it": "Peaking time [µs]",
        "en": "Peaking time [µs]",
    },
    "help_tpeak": {
        "it": "STIMA: valore tipico; quello vero è TPEA= nel .mca (sezione DP5 CONFIGURATION).",
        "en": "ESTIMATE: typical value; the actual one is TPEA= in the .mca (DP5 CONFIGURATION section).",
    },
    "label_tau_ratio": {
        "it": "τ / Tpeak",
        "en": "τ / Tpeak",
    },
    "help_tau": {
        "it": "Dalla scheda FAST SDD: Tpeak 9.6 µs → 18 kcps in uscita al 50 % di dead time; "
             "con modello paralizzabile τ ≈ 2.0 × Tpeak.",
        "en": "From the FAST SDD data sheet: Tpeak 9.6 µs → 18 kcps output at 50 % dead time; "
             "with a paralyzable model τ ≈ 2.0 × Tpeak.",
    },
    "err_invalid_comp": {
        "it": "Composizione non valida: {exc}",
        "en": "Invalid composition: {exc}",
    },
    "err_comp_sum_zero": {
        "it": "composizione con somma nulla",
        "en": "composition with zero sum",
    },
    "caption_si_mismatch": {
        "it": "⚠️ Si attivo {si:g} µm ≠ lab_xrf.yml ({cfg:g} µm)",
        "en": "⚠️ Active Si {si:g} µm ≠ lab_xrf.yml ({cfg:g} µm)",
    },
    "caption_si_mismatch_geant4": {
        "it": " — con la matrice Geant4 lo spessore deve essere quello della simulazione MC.",
        "en": " — with the Geant4 matrix the thickness must match that of the MC simulation.",
    },

    # composizione
    "exp_composition": {
        "it": "🧪 Composizione del campione simulato",
        "en": "🧪 Simulated sample composition",
    },
    "col_el": {
        "it": "El",
        "en": "El",
    },
    "col_wt": {
        "it": "wt %",
        "en": "wt %",
    },
    "col_ppm": {
        "it": "ppm",
        "en": "ppm",
    },
    "caption_comp_norm": {
        "it": "Frazioni normalizzate a somma 1 (usate anche per tabelle e LLD).",
        "en": "Fractions normalized to sum 1 (also used for the tables and the LLD).",
    },

    # spettro misurato
    "exp_measured": {
        "it": "📥 Spettro misurato (.mca Amptek/PMCA) — sovrapponi e confronta",
        "en": "📥 Measured spectrum (.mca Amptek/PMCA) — overlay and compare",
    },
    "label_mca_file": {
        "it": "File .mca",
        "en": ".mca file",
    },
    "err_invalid_file": {
        "it": "File non valido: {exc}",
        "en": "Invalid file: {exc}",
    },
    "err_read_file": {
        "it": "Errore nella lettura del file ({typ}): {exc}",
        "en": "Error reading the file ({typ}): {exc}",
    },
    "warn_cal_note": {
        "it": "Calibrazione: {note}. Inserisci A, B a mano.",
        "en": "Calibration: {note}. Enter A and B manually.",
    },
    "warn_no_cal": {
        "it": "Nessuna sezione <<CALIBRATION>>: inserisci A, B a mano.",
        "en": "No <<CALIBRATION>> section: enter A and B manually.",
    },
    "label_cal_a": {
        "it": "A [keV]",
        "en": "A [keV]",
    },
    "label_cal_b": {
        "it": "B [keV/canale]",
        "en": "B [keV/channel]",
    },
    "label_cal_c": {
        "it": "C [keV/canale²]",
        "en": "C [keV/channel²]",
    },
    "warn_live_missing": {
        "it": "LIVE_TIME mancante o nullo nel file: inserisci il live time a mano{real}",
        "en": "LIVE_TIME missing or zero in the file: enter the live time manually{real}",
    },
    "warn_live_missing_real": {
        "it": " (REAL_TIME = {real:g} s).",
        "en": " (REAL_TIME = {real:g} s).",
    },
    "warn_live_missing_end": {
        "it": ".",
        "en": ".",
    },
    "label_meas_live": {
        "it": "Live time misurato [s]",
        "en": "Measured live time [s]",
    },
    "toggle_align": {
        "it": "Simula sui canali del misurato",
        "en": "Simulate on the measured channels",
    },
    "label_norm": {
        "it": "Normalizzazione",
        "en": "Normalization",
    },
    "norm_none": {
        "it": "nessuna",
        "en": "none",
    },
    "norm_live": {
        "it": "live time",
        "en": "live time",
    },
    "norm_peak": {
        "it": "su un picco",
        "en": "to a peak",
    },
    "cal_from_file": {
        "it": "dal file ({label}, unità {unit}, {deg})",
        "en": "from the file ({label}, unit {unit}, {deg})",
    },
    "cal_no_label": {
        "it": "senza etichetta",
        "en": "unlabelled",
    },
    "cal_quadratic": {
        "it": "quadratica",
        "en": "quadratic",
    },
    "cal_linear": {
        "it": "lineare",
        "en": "linear",
    },
    "cal_manual": {
        "it": "manuale",
        "en": "manual",
    },
    "caption_meas": {
        "it": "{n} canali · LIVE_TIME {live:g} s · REAL_TIME {real} s · "
             "calibrazione {cal} · conteggi totali {tot:,.0f}",
        "en": "{n} channels · LIVE_TIME {live:g} s · REAL_TIME {real} s · "
             "calibration {cal} · total counts {tot:,.0f}",
    },
    "info_tpea": {
        "it": "Peaking time TPEA [µs]",
        "en": "Peaking time TPEA [µs]",
    },
    "info_gain": {
        "it": "Gain (GAIN)",
        "en": "Gain (GAIN)",
    },
    "info_fast": {
        "it": "Fast count",
        "en": "Fast count",
    },
    "info_slow": {
        "it": "Slow count",
        "en": "Slow count",
    },
    "info_meas_dt": {
        "it": "Dead time misurato [%]",
        "en": "Measured dead time [%]",
    },
    "popover_dp5": {
        "it": "ℹ️ Impostazioni DP5 e stato DPP dal file",
        "en": "ℹ️ DP5 settings and DPP status from the file",
    },
    "col_item": {
        "it": "voce",
        "en": "item",
    },
    "col_value": {
        "it": "valore",
        "en": "value",
    },
    "col_key": {
        "it": "chiave",
        "en": "key",
    },
    "ch_src_measured": {
        "it": "canali del misurato",
        "en": "measured channels",
    },
    "ch_src_config": {
        "it": "calibrazione MCA della config",
        "en": "config MCA calibration",
    },
    "ch_src_grid": {
        "it": "griglia di simulazione ({step:.0f} eV/canale)",
        "en": "simulation grid ({step:.0f} eV/channel)",
    },

    # metriche e avvisi
    "spinner": {
        "it": "Simulazione FP in corso — {sample}, {kv} kV, griglia {n}×{n}…",
        "en": "FP simulation in progress — {sample}, {kv} kV, grid {n}×{n}…",
    },
    "err_sim_failed": {
        "it": "La simulazione è fallita.",
        "en": "The simulation failed.",
    },
    "metric_total_counts": {
        "it": "Conteggi totali",
        "en": "Total counts",
    },
    "help_total_counts": {
        "it": "Nel live time di {lt:g} s, {poisson} Poisson",
        "en": "In a live time of {lt:g} s, {poisson} Poisson",
    },
    "with_poisson": {
        "it": "con",
        "en": "with",
    },
    "without_poisson": {
        "it": "senza",
        "en": "without",
    },
    "metric_rate": {
        "it": "Rate rivelato",
        "en": "Detected rate",
    },
    "help_rate": {
        "it": "Integrale dello spettro rivelato (sopra ~1 keV)",
        "en": "Integral of the detected spectrum (above ~1 keV)",
    },
    "metric_dead": {
        "it": "Dead time (STIMA)",
        "en": "Dead time (ESTIMATE)",
    },
    "help_dead": {
        "it": "1−exp(−R·τ), τ = {tau:.2g} µs ({ratio:g} × Tpeak {tpeak:g} µs); "
             "real time stimato {rt:.0f} s. Ar, righe della box e pile-up non inclusi.",
        "en": "1−exp(−R·τ), τ = {tau:.2g} µs ({ratio:g} × Tpeak {tpeak:g} µs); "
             "estimated real time {rt:.0f} s. Ar, box lines and pile-up not included.",
    },
    "metric_power": {
        "it": "Potenza tubo",
        "en": "Tube power",
    },
    "metric_power_max": {
        "it": "max {w:.0f} W",
        "en": "max {w:.0f} W",
    },
    "metric_fwhm": {
        "it": "FWHM @ 5.9 keV",
        "en": "FWHM @ 5.9 keV",
    },
    "metric_compute": {
        "it": "Tempo di calcolo",
        "en": "Compute time",
    },
    "help_compute": {
        "it": "Primo run per kV/filtro/target: +~7 s per SpekPy. Poi cache.",
        "en": "First run for a given kV/filter/target: +~7 s for SpekPy. Then cached.",
    },
    "err_power": {
        "it": "⚠️ {kv} kV × {uA} µA = {power:.2f} W: oltre i 4 W del Mini-X2. "
             "A {kv} kV la corrente massima è {imax:.0f} µA.",
        "en": "⚠️ {kv} kV × {uA} µA = {power:.2f} W: exceeds the Mini-X2 4 W limit. "
             "At {kv} kV the maximum current is {imax:.0f} µA.",
    },
    "warn_cells": {
        "it": "Solo {n_valid} celle su {n_tot} sono viste da tubo e rivelatore.",
        "en": "Only {n_valid} of {n_tot} cells are seen by both the tube and the detector.",
    },
    "caption_dt_meas": {
        "it": "Dead time misurato dal file: {meas:.1f} % ({src}) · stimato per la simulazione: {est:.1f} %",
        "en": "Dead time measured from the file: {meas:.1f} % ({src}) · estimated for the simulation: {est:.1f} %",
    },
    "dt_fast_slow": {
        "it": "fast/slow count",
        "en": "fast/slow count",
    },
    "dt_live_real": {
        "it": "1 − LIVE/REAL",
        "en": "1 − LIVE/REAL",
    },

    # tab
    "tab_spectrum": {
        "it": "📈 Spettro",
        "en": "📈 Spectrum",
    },
    "tab_lines": {
        "it": "📋 Righe & LLD",
        "en": "📋 Lines & LLD",
    },
    "tab_compare": {
        "it": "🔀 Confronto run",
        "en": "🔀 Run comparison",
    },
    "tab_instrument": {
        "it": "🛰️ Strumento",
        "en": "🛰️ Instrument",
    },
    "tab_notes": {
        "it": "📝 Note fisiche",
        "en": "📝 Physical notes",
    },

    # tab spettro
    "label_yaxis": {
        "it": "Asse y",
        "en": "Y axis",
    },
    "y_counts": {
        "it": "conteggi/canale",
        "en": "counts/channel",
    },
    "y_density": {
        "it": "ph/s/keV",
        "en": "ph/s/keV",
    },
    "label_scale": {
        "it": "Scala",
        "en": "Scale",
    },
    "scale_log": {
        "it": "log",
        "en": "log",
    },
    "scale_lin": {
        "it": "lin",
        "en": "linear",
    },
    "label_energy": {
        "it": "Energia [keV]",
        "en": "Energy [keV]",
    },
    "label_components": {
        "it": "Componenti",
        "en": "Components",
    },
    "comp_fluo": {
        "it": "fluorescenza",
        "en": "fluorescence",
    },
    "comp_coh": {
        "it": "coerente",
        "en": "coherent",
    },
    "comp_inc": {
        "it": "incoerente",
        "en": "incoherent",
    },
    "help_components": {
        "it": "Con la matrice Geant4 le componenti non sono separabili.",
        "en": "With the Geant4 matrix the components cannot be separated.",
    },
    "toggle_labels": {
        "it": "Etichette righe",
        "en": "Line labels",
    },
    "toggle_ar": {
        "it": "Zona Ar",
        "en": "Ar region",
    },
    "toggle_expected": {
        "it": "Valore atteso",
        "en": "Expected value",
    },
    "label_n_labels": {
        "it": "Numero massimo di etichette",
        "en": "Maximum number of labels",
    },
    "trace_sim_poisson": {
        "it": "simulato + Poisson (seed {seed})",
        "en": "simulated + Poisson (seed {seed})",
    },
    "trace_sim_expected": {
        "it": "simulato (atteso)",
        "en": "simulated (expected)",
    },
    "trace_geant4_suffix": {
        "it": " · Geant4",
        "en": " · Geant4",
    },
    "label_norm_peak": {
        "it": "Picco per la normalizzazione",
        "en": "Peak for normalization",
    },
    "trace_measured": {
        "it": "misurato × {k:.3g}",
        "en": "measured × {k:.3g}",
    },
    "trace_residuals": {
        "it": "(mis − sim)/√sim",
        "en": "(meas − sim)/√sim",
    },
    "axis_residuals": {
        "it": "residui [σ]",
        "en": "residuals [σ]",
    },
    "anno_residuals": {
        "it": "Residui disponibili con «Simula sui canali del misurato»",
        "en": "Residuals available with “Simulate on the measured channels”",
    },
    "anno_ar": {
        "it": "Ar K (2.96 keV): non modellato",
        "en": "Ar K (2.96 keV): not modelled",
    },
    "axis_counts": {
        "it": "conteggi / canale",
        "en": "counts / channel",
    },
    "axis_density": {
        "it": "ph / s / keV",
        "en": "ph / s / keV",
    },
    "axis_energy": {
        "it": "Energia [keV]",
        "en": "Energy [keV]",
    },
    "plot_title": {
        "it": "<b>{sample}</b> · {kv} kV · {uA} µA · {atm} · LT {lt:g} s · griglia {n}×{n}",
        "en": "<b>{sample}</b> · {kv} kV · {uA} µA · {atm} · LT {lt:g} s · grid {n}×{n}",
    },
    "caption_channels": {
        "it": "Canali: {src} · {n} canali · picchi integrati sui canali "
             "(continuo con interpolazione monotona della cumulata) · angolo di scattering {psi:.0f}° · "
             "spettro del tubo da {emin:.3f} keV.",
        "en": "Channels: {src} · {n} channels · peaks integrated onto the channels "
             "(continuum with monotonic interpolation of the cumulative) · scattering angle {psi:.0f}° · "
             "tube spectrum from {emin:.3f} keV.",
    },

    # esportazione
    "section_export": {
        "it": "#### 💾 Salva ed esporta",
        "en": "#### 💾 Save and export",
    },
    "label_run": {
        "it": "Etichetta run",
        "en": "Run label",
    },
    "btn_add_compare": {
        "it": "📌 Aggiungi al confronto",
        "en": "📌 Add to comparison",
    },
    "toast_run_saved": {
        "it": "Run salvato: {label}",
        "en": "Run saved: {label}",
    },
    "help_mca": {
        "it": "REAL_TIME = LIVE_TIME/(1−DT) = {rt:.1f} s (dead time stimato)",
        "en": "REAL_TIME = LIVE_TIME/(1−DT) = {rt:.1f} s (estimated dead time)",
    },
    "csv_comment_channels": {
        "it": "# {desc}; live_time_s={lt}; canali: {src}\n",
        "en": "# {desc}; live_time_s={lt}; channels: {src}\n",
    },
    "csv_comment_grid": {
        "it": "\n# Densità sulla griglia di simulazione (ph/s/keV, risposta gaussiana)\n",
        "en": "\n# Density on the simulation grid (ph/s/keV, Gaussian response)\n",
    },
    "sheet_lines": {
        "it": "Righe",
        "en": "Lines",
    },
    "sheet_lld": {
        "it": "LLD (stima)",
        "en": "LLD (estimate)",
    },
    "sheet_params": {
        "it": "Parametri",
        "en": "Parameters",
    },
    "col_param": {
        "it": "parametro",
        "en": "parameter",
    },
    "col_param_value": {
        "it": "valore",
        "en": "value",
    },
    "help_config_zip": {
        "it": "Config + composizione caricata (compositions_gui/).",
        "en": "Config + uploaded composition (compositions_gui/).",
    },
    "poisson_seed": {
        "it": "Poisson seed {seed}",
        "en": "Poisson seed {seed}",
    },
    "expected_word": {
        "it": "atteso",
        "en": "expected",
    },
    "dt_est_tag": {
        "it": "DT_stima",
        "en": "DT_estimate",
    },

    # tab righe
    "warn_geant4_lines": {
        "it": "Matrice Geant4 attiva: lo spettro usa la matrice, ma tabella righe, P/B e LLD usano picchi "
             "gaussiani e QE solo fotoelettrica (approssimazione).",
        "en": "Geant4 matrix active: the spectrum uses the matrix, but the line table, P/B and LLD use Gaussian "
             "peaks and photoelectric-only QE (approximation).",
    },
    "section_fluo_lines": {
        "it": "#### Righe di fluorescenza",
        "en": "#### Fluorescence lines",
    },
    "col_element": {
        "it": "Elemento",
        "en": "Element",
    },
    "col_line": {
        "it": "Riga",
        "en": "Line",
    },
    "col_e_kev": {
        "it": "E [keV]",
        "en": "E [keV]",
    },
    "col_conc_ppm": {
        "it": "Conc. [ppm]",
        "en": "Conc. [ppm]",
    },
    "col_entrance": {
        "it": "Ingresso riv. [ph/s]",
        "en": "Detector entrance [ph/s]",
    },
    "col_tbe_qe": {
        "it": "T_Be·QE_Si",
        "en": "T_Be·QE_Si",
    },
    "col_detected": {
        "it": "Rivelati [cps]",
        "en": "Detected [cps]",
    },
    "col_counts_in_lt": {
        "it": "Conteggi in {lt:g} s",
        "en": "Counts in {lt:g} s",
    },
    "label_min_cps": {
        "it": "Mostra righe con rate rivelato ≥ [cps]",
        "en": "Show lines with detected rate ≥ [cps]",
    },
    "section_pb_lld": {
        "it": "#### Picco/fondo e limite di rivelabilità  <span class='badge warn'>STIMA</span>",
        "en": "#### Peak-to-background and detection limit (LLD)  <span class='badge warn'>ESTIMATE</span>",
    },
    "label_lld_def": {
        "it": "Definizione LLD",
        "en": "LLD definition",
    },
    "help_lld_def": {
        "it": "3√B: 3√B/N·c. Currie: L_D ≈ 2.71 + 4.65√B conteggi (blank appaiato).",
        "en": "3√B: 3√B/N·c. Currie: L_D ≈ 2.71 + 4.65√B counts (paired blank).",
    },
    "col_line_used": {
        "it": "Riga usata",
        "en": "Line used",
    },
    "col_fwhm_ev": {
        "it": "FWHM [eV]",
        "en": "FWHM [eV]",
    },
    "col_n_peak": {
        "it": "N picco (finestra)",
        "en": "N peak (window)",
    },
    "col_b_bkg": {
        "it": "B fondo (finestra)",
        "en": "B background (window)",
    },
    "col_pb": {
        "it": "P/B",
        "en": "P/B",
    },
    "col_stat_sigma": {
        "it": "σ statistico [%]",
        "en": "Statistical σ [%]",
    },
    "col_lld_ka": {
        "it": "LLD Kα [ppm]",
        "en": "LLD Kα [ppm]",
    },
    "col_lld_la": {
        "it": "LLD Lα [ppm]",
        "en": "LLD Lα [ppm]",
    },
    "col_lld_value": {
        "it": "LLD {definition} [ppm] (STIMA)",
        "en": "LLD {definition} [ppm] (ESTIMATE)",
    },
    "axis_lld": {
        "it": "LLD [ppm]",
        "en": "LLD [ppm]",
    },
    "title_lld": {
        "it": "LLD ({definition}) in {lt:g} s",
        "en": "LLD ({definition}) in {lt:g} s",
    },
    "caption_lld": {
        "it": "Per ogni elemento si valutano Kα e Lα (se eccitate e dentro la griglia) e si sceglie la LLD migliore. "
             "Finestra = {win} × FWHM; N = conteggi netti del gruppo nella finestra; B = continuo di "
             "scattering + code delle altre righe nella finestra. Concentrazioni normalizzate a somma 1, risposta "
             "lineare. Con 3√B e B < 10 conteggi la LLD non è definita (cella vuota). Senza Ar, pile-up, escape e "
             "shelf: ordine di grandezza, non un valore di specifica.",
        "en": "For each element, Kα and Lα are evaluated (if excited and inside the grid) and the better LLD is chosen. "
             "Window = {win} × FWHM; N = net counts of the group in the window; B = scattering continuum "
             "+ tails of the other lines in the window. Concentrations normalized to sum 1, linear response. "
             "With 3√B and B < 10 counts the LLD is undefined (empty cell). Without Ar, pile-up, escape and "
             "shelf: an order of magnitude, not a specification value.",
    },

    # tab confronto
    "info_no_runs": {
        "it": "Nessun run salvato. Nel tab «Spettro» premi **📌 Aggiungi al confronto** (per esempio "
             "una volta in aria e una nel vuoto, oppure con due campioni).",
        "en": "No run saved yet. In the “Spectrum” tab press **📌 Add to comparison** (for example "
             "once in air and once in vacuum, or with two samples).",
    },
    "toggle_norm_ua": {
        "it": "Normalizza per µA",
        "en": "Normalize per µA",
    },
    "btn_clear_compare": {
        "it": "🗑️ Svuota il confronto",
        "en": "🗑️ Clear comparison",
    },
    "axis_density_per_ua": {
        "it": "ph/s/keV per µA",
        "en": "ph/s/keV per µA",
    },
    "section_main_lines": {
        "it": "**Righe principali (rate rivelato)** — ",
        "en": "**Main lines (detected rate)** — ",
    },

    # tab strumento
    "name_tube": {
        "it": "tubo Mini-X2",
        "en": "Mini-X2 tube",
    },
    "name_det": {
        "it": "X-123",
        "en": "X-123",
    },
    "anno_illuminated": {
        "it": "area illuminata ±{hx:.2f} × ±{hy:.2f} cm",
        "en": "illuminated area ±{hx:.2f} × ±{hy:.2f} cm",
    },
    "title_geo": {
        "it": "Geometria (proiezione x–z{yoff}) · scattering {psi:.0f}° · {mode}",
        "en": "Geometry (x–z projection{yoff}) · scattering {psi:.0f}° · {mode}",
    },
    "geo_y_off": {
        "it": " — componenti y non mostrate",
        "en": " — y components not shown",
    },
    "geo_mode_config": {
        "it": "config",
        "en": "config",
    },
    "geo_mode_param": {
        "it": "parametrica",
        "en": "parametric",
    },
    "axis_x_cm": {
        "it": "x [cm]",
        "en": "x [cm]",
    },
    "axis_z_cm": {
        "it": "z [cm]",
        "en": "z [cm]",
    },
    "eff_be": {
        "it": "Be {um:.1f} µm",
        "en": "Be {um:.1f} µm",
    },
    "eff_si": {
        "it": "Si {um:.0f} µm (fotoel.)",
        "en": "Si {um:.0f} µm (photoel.)",
    },
    "eff_atm_path": {
        "it": "{atm} {L:.2f} cm",
        "en": "{atm} {L:.2f} cm",
    },
    "eff_total": {
        "it": "totale",
        "en": "total",
    },
    "title_efficiency": {
        "it": "Efficienza lato rivelatore",
        "en": "Detector-side efficiency",
    },
    "axis_transmission": {
        "it": "trasmissione / efficienza",
        "en": "transmission / efficiency",
    },
    "trace_tube_out": {
        "it": "uscita tubo",
        "en": "tube output",
    },
    "trace_after_air": {
        "it": "dopo {L:.2f} cm d'aria",
        "en": "after {L:.2f} cm of air",
    },
    "title_tube": {
        "it": "Spettro del tubo (SpekPy) · {kv} kV · Ag {um} µm · filtri {filt}",
        "en": "Tube spectrum (SpekPy) · {kv} kV · Ag {um} µm · filters {filt}",
    },
    "axis_tube_y": {
        "it": "ph/s/keV/sr",
        "en": "ph/s/keV/sr",
    },
    "warn_tube": {
        "it": "Spettro del tubo non disponibile: {exc}",
        "en": "Tube spectrum not available: {exc}",
    },
    "title_comp": {
        "it": "Composizione normalizzata · {sample}",
        "en": "Normalized composition · {sample}",
    },
    "axis_ppm_mass": {
        "it": "ppm (massa)",
        "en": "ppm (mass)",
    },
    "exp_config_log": {
        "it": "🧾 Config usata (YAML) e log del pipeline",
        "en": "🧾 Config used (YAML) and pipeline log",
    },
    "log_cache": {
        "it": "(nessun output: risultato dalla cache)",
        "en": "(no output: result from cache)",
    },

    # parse_mca / export
    "err_mca_no_data": {
        "it": "Il file non contiene la sezione <<DATA>>: non sembra un .mca PMCA/Amptek.",
        "en": "The file has no <<DATA>> section: it does not look like a PMCA/Amptek .mca file.",
    },
    "err_mca_non_numeric": {
        "it": "Valore non numerico nella sezione <<DATA>>: {val}",
        "en": "Non-numeric value in the <<DATA>> section: {val}",
    },
    "err_mca_few_channels": {
        "it": "La sezione <<DATA>> ha solo {n} canali.",
        "en": "The <<DATA>> section has only {n} channels.",
    },
    "cal_note_unrecognized": {
        "it": "sezione <<CALIBRATION>> non riconosciuta",
        "en": "unrecognized <<CALIBRATION>> section",
    },
    "cal_note_bad_b": {
        "it": "calibrazione scartata: B = {b:.4g} eV/canale fuori dall'intervallo fisico "
             "{lo:g}–{hi:g} eV/canale (unità dedotte: {unit})",
        "en": "calibration discarded: B = {b:.4g} eV/channel outside the physical range "
             "{lo:g}–{hi:g} eV/channel (inferred unit: {unit})",
    },
    "export_head_date": {
        "it": "# Config esportata dalla GUI XRF il {date}",
        "en": "# Config exported from the XRF GUI on {date}",
    },
    "export_head_base": {
        "it": "# Base: configs/lab_xrf.yml — tube.current_mA e mca_live_time_s = valori della GUI.",
        "en": "# Base: configs/lab_xrf.yml — tube.current_mA and mca_live_time_s = GUI values.",
    },
    "export_head_poisson": {
        "it": "# Rumore di Poisson e canali MCA sono calcolati dalla GUI, non dal pipeline.",
        "en": "# Poisson noise and MCA channels are computed by the GUI, not by the pipeline.",
    },
    "export_head_removed": {
        "it": "# Chiavi di lab_xrf.yml rimosse in questa config: {keys} "
             "(file non disponibili o canali gestiti dalla GUI).",
        "en": "# Keys from lab_xrf.yml removed in this config: {keys} "
             "(files not available, or channels handled by the GUI).",
    },
    "export_head_comp": {
        "it": "# Composizione caricata: {path} (inclusa nello zip).",
        "en": "# Uploaded composition: {path} (included in the zip).",
    },
    "export_comp_notes": {
        "it": "composizione caricata nella GUI",
        "en": "composition uploaded in the GUI",
    },

    # note fisiche (markdown intero)
    "tag_filters": {
        "it": "filtri",
        "en": "filters",
    },
    "xlsx_suffix": {
        "it": "_righe.xlsx",
        "en": "_lines.xlsx",
    },

    "notes_md": {
        "it": """
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
6. **Inizio griglia a {e_min:.3f} keV** (SpekPy): mancano solo le righe sotto ~1 keV (O, C, …). Na Kα (1.04 keV)
   è al limite della griglia; Mg Kα (1.25 keV) c'è, ma in aria è fortemente assorbita.
7. **Dead time** stimato con modello paralizzabile, τ = (τ/Tpeak) × Tpeak; il peaking time di default (4.8 µs) è una
   STIMA. Il `.mca` esportato ha REAL_TIME = LIVE_TIME/(1 − DT). Pile-up e somme non modellati.
8. **Geometria:** in modalità parametrica è nel piano x–z con il rivelatore puntato sullo spot (`det_normal = −r_det`);
   in modalità config si usano i vettori di `lab_xrf.yml` così come sono.
9. **Etichette Ag K** (Rayleigh e Compton) mostrate solo sopra il bordo K dell'Ag ({ag_edge} keV).

#### Unità
- Tabella righe: *ingresso rivelatore* = ph/s che arrivano al rivelatore (geometria + aria), *rivelati* = × T_Be × QE_Si.
- Concentrazioni: frazioni in massa normalizzate a somma 1.
- Spettro: ph/s/keV (densità) oppure conteggi per canale nel live time scelto.
""",
        "en": """
#### What the `xrf_model` package does and what the app adds
- **Physics (`xrf_model`, unchanged):** primary K fluorescence (Sherman/Shiraiwa–Fujino) and secondary fluorescence,
  L lines with Coster–Kronig and K→L cascade, Rayleigh with f0+f′, Compton Klein–Nishina × S with Doppler
  broadening, air absorption, Be window and Si efficiency, Fano + noise resolution. Tube spectrum from SpekPy.
- **App:** the simulation is run at 1 µA and scaled to the selected tube current (the model is linear in current:
  no dead time or pile-up). Gaussian peaks are integrated analytically onto the destination MCA channels;
  the continuum is rebinned with a monotonic interpolation of the cumulative (total conserved). **Poisson
  noise** is sampled on the final channels (`numpy.random.default_rng(seed).poisson`). P/B, LLD and dead time are estimates.

#### Known limitations
1. **Argon is not modelled.** Air absorbs but does not emit or scatter: the Ar K line (2.96 keV) and air
   scattering are missing, so the 2.8–3.4 keV region cannot be compared with a measurement in air.
2. **No footprint model.** The grid (±0.4 cm in the config) is treated as a uniformly illuminated area.
   With a collimator the absolute intensities are overestimated; “Illuminated area → from collimator” uses the formula
   in `lab_xrf.yml` (½x = 0.443·D/cosθ, ½y = 0.443·D).
3. **Ag target:** config 1.0 µm, Amptek data sheet/manual 0.75 ± 0.1 µm. With 0.75 µm the lines drop by ~21–25 %.
4. **Active Si:** app default 500 µm (ORTEC quotation and X-123 manual); `lab_xrf.yml` has {cfg_si:g} µm.
5. **Geant4 response matrix not included in the repository:** without the `.npz` the peaks are pure Gaussians, with no escape
   (1.74 keV below each peak), tail or shelf. The MCA calibration file (`.calib`) is also not in the repository.
6. **Grid starts at {e_min:.3f} keV** (SpekPy): only lines below ~1 keV are missing (O, C, …). Na Kα (1.04 keV)
   is at the edge of the grid; Mg Kα (1.25 keV) is present, but in air it is strongly absorbed.
7. **Dead time** estimated with a paralyzable model, τ = (τ/Tpeak) × Tpeak; the default peaking time (4.8 µs) is an
   ESTIMATE. The exported `.mca` has REAL_TIME = LIVE_TIME/(1 − DT). Pile-up and sum peaks are not modelled.
8. **Geometry:** in parametric mode it lies in the x–z plane with the detector aimed at the spot (`det_normal = −r_det`);
   in config mode the vectors from `lab_xrf.yml` are used as they are.
9. **Ag K labels** (Rayleigh and Compton) are shown only above the Ag K edge ({ag_edge} keV).

#### Units
- Line table: *detector entrance* = ph/s arriving at the detector (geometry + air), *detected* = × T_Be × QE_Si.
- Concentrations: mass fractions normalized to sum 1.
- Spectrum: ph/s/keV (density) or counts per channel in the chosen live time.
""",
    },
}
