"""Test funzionale della GUI XRF.

Uso (dalla root del repo, con app.py e xrf_gui_core.py accanto a src/):
    python test_functional.py
Importa le funzioni di calcolo da xrf_gui_core e le confronta con run_one_grid.
Con streamlit installato verifica anche l'app (AppTest), l'accesso con password
e il cambio lingua IT/EN.
"""
from __future__ import annotations

import contextlib
import copy
import io
import logging
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
logging.disable(logging.WARNING)

import ast                              # noqa: E402
import re                               # noqa: E402
import numpy as np                      # noqa: E402
import i18n                             # noqa: E402
import xrf_gui_core as core             # noqa: E402

i18n.set_lang("it")

FAILS = []


def check(name, cond, detail=""):
    print(f"[{'OK ' if cond else 'NO '}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


# 0. i18n: ogni chiave ha it e en, stessi placeholder ──────────────────────
probs = i18n.translation_problems()
check("i18n: ogni chiave ha it e en (placeholder allineati)", not probs,
      f"{len(probs)} problemi" + (": " + "; ".join(probs[:8]) if probs else ""))
check("i18n: dizionario non vuoto", len(i18n.STRINGS) >= 200, f"{len(i18n.STRINGS)} chiavi")

cfg0 = core.base_config()
g = core.geometry_defaults(cfg0)
P = dict(sample="soil_14163", custom_comp=None, secondary=True, kv=50, target_um=1.0,
         keep_filters=sorted(cfg0["tube"].get("filters", {})), filter_mat="nessuno", filter_mm=0.0,
         geo_mode="parametrica", theta_src=g["theta_src"], D_src=g["D_src"], theta_det=g["theta_det"],
         D_det=g["D_det"], det_area_mm2=17.0, atmosphere="air", src_vac=0.0, det_vac=0.0,
         grid_hx=0.4, grid_hy=0.4, grid_n=2, be_um=12.7, si_um=450.0, fano=0.118, w_ev=3.65,
         noise_ev=12.7, use_response=False)

# 1. coincidenza con run_one_grid (aria e vuoto) ───────────────────────────────
sims = {}
for atm in ("air", "vacuum"):
    p = dict(P, atmosphere=atm)
    sim = core.scaled(core.simulate_core(p, cfg0), 2.0)          # 2 µA = corrente di lab_xrf.yml
    sims[atm] = sim
    cfg = copy.deepcopy(cfg0)
    cfg["atmosphere"] = atm
    for k in core.REMOVED_KEYS:
        cfg.pop(k, None)
    cfg["paths"]["output_dir"] = tempfile.mkdtemp()
    with contextlib.redirect_stdout(io.StringIO()):
        ref = core.run_one_grid(None, cfg)["soil_14163"]
    d1 = sim["fluo"].set_index(["Element", "Line"])["Intensity_ph_s"]
    d2 = ref["fluorescence"].set_index(["Element", "Line"])["Intensity_ph_s"]
    rel_lines = float(np.max(np.abs(d1 - d2) / np.maximum(d2, 1e-30)))
    rel_spec = float(np.max(np.abs(sim["spec"] - ref["spectrum"])) / np.max(ref["spectrum"]))
    gsum = sim["comp_fluo"] + sim["comp_coh"] + sim["comp_inc"]
    rel_comp = float(np.max(np.abs(gsum - ref["spectrum"])) / np.max(ref["spectrum"]))
    check(f"[{atm}] righe e spettro = run_one_grid", rel_lines < 1e-9 and rel_spec < 1e-9 and rel_comp < 1e-9,
          f"righe {rel_lines:.1e}, spettro {rel_spec:.1e}, componenti {rel_comp:.1e}; "
          f"Fe Kα1 {d1[('Fe', 'Kα1')]:.1f} ph/s, Si Kα1 {d1[('Si', 'Kα1')]:.3f} ph/s")
sim = sims["air"]

# 2. normalizzazione composizione: frazioni vs wt% → stessa LLD ─────────────
comp = core.normalized(core.load_composition(core.composition_path(cfg0, "soil_14163")))
lld = {}
for name, scale in (("frazioni", 1.0), ("wt%", 100.0)):
    p = dict(P, sample=f"test_{name.replace('%', 'pct')}", custom_comp={k: v * scale for k, v in comp.items()})
    s = core.scaled(core.simulate_core(p, cfg0), 2.0)
    t = core.lld_table(s, 300.0, 1.0, 50.0).set_index("Elemento")
    lld[name] = t
a, b = lld["frazioni"].loc["Sr", "LLD 3√B [ppm] (STIMA)"], lld["wt%"].loc["Sr", "LLD 3√B [ppm] (STIMA)"]
ca, cb = lld["frazioni"].loc["Sr", "Conc. [ppm]"], lld["wt%"].loc["Sr", "Conc. [ppm]"]
check("composizione normalizzata (frazioni = wt%)", abs(a / b - 1) < 1e-9 and abs(ca / cb - 1) < 1e-9 and abs(ca - 1e6 * comp["Sr"]) < 1e-6,
      f"LLD Sr {a:.3g} vs {b:.3g} ppm; conc Sr {ca:.1f} ppm")

# 3. residui con live time diversi ─────────────────────────────────────────
edges = core.edges_from_calib((0.0, 0.01, 0.0), 2500)
rate = core.rate_on_channels(sim, edges)
lt_sim, lt_meas = 300.0, 60.0
k_live = lt_sim / lt_meas
sel = rate * lt_meas > 20
res = np.concatenate([core.residuals(core.poisson_counts(rate * lt_meas, sd), rate * lt_sim, k_live)[sel]
                      for sd in range(30)])
res_old = np.concatenate([((core.poisson_counts(rate * lt_meas, sd) * k_live - rate * lt_sim)
                           / np.sqrt(rate * lt_sim))[sel] for sd in range(30)])
check("residui con live time diversi ~ N(0,1)", abs(res.mean()) < 0.05 and abs(res.std() - 1) < 0.05,
      f"media {res.mean():.3f}, std {res.std():.3f} (k = {k_live:g}; vecchia formula std {res_old.std():.2f})")

# 4. filtri del config preservati + filtro aggiunto ─────────────────────────
base2 = copy.deepcopy(cfg0)
base2["tube"]["filters"] = {"Be": 0.125, "Cu": 0.05, "Al": 0.1}
c1 = core.build_cfg(dict(P, keep_filters=None, filter_mat="Ti", filter_mm=0.025), base2)
c2 = core.build_cfg(dict(P, keep_filters=["Be"], filter_mat="Al", filter_mm=0.2), base2)
check("filtri del config preservati e filtro aggiunto",
      c1["tube"]["filters"] == {"Be": 0.125, "Cu": 0.05, "Al": 0.1, "Ti": 0.025}
      and c2["tube"]["filters"] == {"Be": 0.125, "Al": 0.2},
      f"{c1['tube']['filters']} | {c2['tube']['filters']}")

# 5. parsing .mca ─────────────────────────────────────────────────────────
data = "\n".join(str(i % 7) for i in range(1024))
mk = lambda cal: (f"<<PMCA SPECTRUM>>\nLIVE_TIME - 120.5\nREAL_TIME - 130\n{cal}<<DATA>>\n{data}\n<<END>>\n"
                  "<<DP5 CONFIGURATION>>\nTPEA=4.800;    Peaking Time\nGAIN=7.500;\n<<DP5 CONFIGURATION END>>\n"
                  "<<DPP STATUS>>\nFast Count: 100000\nSlow Count: 90000\nReal Time: 130.0\n<<DPP STATUS END>>\n").encode()
m_kev = core.parse_mca(mk("<<CALIBRATION>>\nLABEL - keV\n100 0.6\n1000 6.0\n"))
m_ev = core.parse_mca(mk("<<CALIBRATION>>\nLABEL - Channel\n100 600\n1000 6000\n"))
m_q = core.parse_mca(mk("<<CALIBRATION>>\nLABEL - keV\n100 0.6\n500 3.05\n1000 6.2\n"))
m_bad = core.parse_mca(mk("<<CALIBRATION>>\nLABEL - keV\n100 60\n1000 6000\n"))
ok_q = abs(m_q["calib"][2]) > 1e-8 and m_q["cal_deg"] == 2 and all(
    abs(core.channel_energies(m_q["calib"], 0, np.array([c]))[0] - e) < 1e-9 for c, e in [(100, 0.6), (500, 3.05), (1000, 6.2)])
check("calibrazione keV e eV", abs(m_kev["calib"][1] - 0.006) < 1e-12 and abs(m_ev["calib"][1] - 0.006) < 1e-12
      and m_ev["cal_unit"] == "eV", f"B = {m_kev['calib'][1]}, {m_ev['calib'][1]} keV/ch")
check("calibrazione quadratica con 3 coppie", ok_q, f"C = {m_q['calib'][2]:.3e}")
check("B fuori range → calibrazione scartata con messaggio", m_bad["calib"] is None and "fuori" in m_bad["cal_note"])
check("DP5/DPP: TPEA, GAIN, fast/slow, dead time",
      m_kev["tpeak_us"] == 4.8 and m_kev["gain"] == 7.5 and abs(m_kev["meas_dt"] - 0.1) < 1e-12 and m_kev["live"] == 120.5)
m_nolt = core.parse_mca(b"<<PMCA SPECTRUM>>\nLIVE_TIME - 0\n<<DATA>>\n" + data.encode() + b"\n<<END>>\n")
check("live time nullo segnalato", m_nolt["live_missing"] and m_nolt["live"] is None)
try:
    core.parse_mca(b"1\n2\n3\n")
    check("file senza <<DATA>> → errore", False)
except core.McaError as exc:
    check("file senza <<DATA>> → errore chiaro", "<<DATA>>" in str(exc), str(exc))
rt = core.parse_mca(core.write_mca(np.arange(100), (0.01, 0.02, 0.0), 300.0, 333.3, "test"))
check("scrittura/rilettura .mca", rt["counts"].sum() == 4950 and abs(rt["calib"][1] - 0.02) < 1e-12 and rt["real"] == 333.3)

# 6. broadening sui canali: conservazione e niente scalini ────────────────
E = sim["sE"]
lo_e, hi_e = 1.2, 40.0
fine = core.edges_from_calib((lo_e, 0.0056, 0.0), int((hi_e - lo_e) / 0.0056))
coarse = core.edges_from_calib((lo_e, 0.03, 0.0), int((hi_e - lo_e) / 0.03))
tot_ref = float(core.rebin(E, sim["spec"], np.array([fine[0], fine[-1]]))[0])
tot_f = float(core.rate_on_channels(sim, fine).sum())
tot_c = float(core.rate_on_channels(sim, coarse).sum())
r = core.rate_on_channels(sim, fine)
cf = 0.5 * (fine[1:] + fine[:-1])
w = (cf > 6.30) & (cf < 6.50)
steps = int(np.sum(np.diff(r[w]) == 0))
check("broadening sui canali conserva i conteggi (< 0.5 %)",
      abs(tot_f / tot_ref - 1) < 5e-3 and abs(tot_c / tot_ref - 1) < 5e-3,
      f"fine {tot_f/tot_ref-1:+.2e}, 30 eV {tot_c/tot_ref-1:+.2e}")
check("picco Fe Kα senza scalini su canali da 5.6 eV", steps == 0, f"canali uguali consecutivi: {steps}")

# 7. LLD: Ba valuta K e L e sceglie la migliore ────────────────────────────
t = core.lld_table(sim, 300.0, 1.0, 50.0).set_index("Elemento")
ba = t.loc["Ba"]
best = np.nanmin([ba["LLD Kα [ppm]"], ba["LLD Lα [ppm]"]])
check("LLD Ba: Kα e Lα valutate, scelta la migliore",
      np.isfinite(ba["LLD Kα [ppm]"]) and np.isfinite(ba["LLD Lα [ppm]"]) and abs(ba["LLD 3√B [ppm] (STIMA)"] - best) < 1e-12,
      f"Kα {ba['LLD Kα [ppm]']:.3g}, Lα {ba['LLD Lα [ppm]']:.3g} → {ba['Riga usata']}")
tc = core.lld_table(sim, 300.0, 1.0, 50.0, "Currie").set_index("Elemento")
check("LLD Currie > 3√B (Sr)", tc.loc["Sr", "LLD Currie [ppm] (STIMA)"] > t.loc["Sr", "LLD 3√B [ppm] (STIMA)"],
      f"Sr {t.loc['Sr', 'LLD 3√B [ppm] (STIMA)']:.3g} → {tc.loc['Sr', 'LLD Currie [ppm] (STIMA)']:.3g} ppm")

# 8. Poisson, cache, password ──────────────────────────────────────────────
x = core.poisson_counts(np.full(200000, 37.0), 1)
check("Poisson media/varianza", abs(x.mean() - 37) < 0.2 and abs(x.var() - 37) < 0.6, f"{x.mean():.2f}/{x.var():.2f}")
k1 = core.phys_key(P, cfg0)
check("chiave cache include lab_xrf.yml e composizione", '"_cfg_stamp"' in k1 and '"_comp_stamp"' in k1)
check("password: confronto",
      core.check_password("giusta", "giusta") and not core.check_password("sbagliata", "giusta")
      and not core.check_password(None, "giusta") and core.check_password(None, None))
ok_geo, _ = core.geometry_is_parametric(cfg0)
bad = dict(cfg0, det_y=1.0)
check("geometria del config riconosciuta", ok_geo and not core.geometry_is_parametric(bad)[0])

# 9. app headless (AppTest) e accesso con password ─────────────────────────
try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None
if AppTest is not None:
    app = str(HERE / "app.py")
    os.environ.pop("XRF_APP_PASSWORD", None)
    at = AppTest.from_file(str(HERE / "app.py"), default_timeout=300).run()
    check("app senza password: parte e simula", not at.exception and len(at.metric) == 6,
          f"eccezioni {len(at.exception)}, metriche {len(at.metric)}")
    os.environ["XRF_APP_PASSWORD"] = "segreta-di-prova"
    at = AppTest.from_file(str(HERE / "app.py"), default_timeout=300).run()
    blocked = len(at.metric) == 0 and len(at.text_input) == 1
    at.text_input[0].input("sbagliata")
    at.get("form_submit_button")[0].click()
    at.run()
    still = len(at.metric) == 0 and any("errata" in e.value for e in at.error)
    at.text_input[0].input("segreta-di-prova")
    at.get("form_submit_button")[0].click()
    at.run()
    passed = not at.exception and len(at.metric) == 6
    check("password: blocca senza, rifiuta la sbagliata, entra con quella giusta", blocked and still and passed,
          f"bloccata {blocked}, rifiutata {still}, entrata {passed}")
    os.environ.pop("XRF_APP_PASSWORD", None)

# 10. testi core in inglese e assenza di italiano hardcodato in app.py ──────
i18n.set_lang("en")
_, reason_en = core.geometry_is_parametric(dict(cfg0, det_y=1.0))
check("geometria: motivo in inglese", "off the x–z plane" in reason_en, reason_en)
m_bad_en = core.parse_mca(mk("<<CALIBRATION>>\nLABEL - keV\n100 60\n1000 6000\n"))
check("calibrazione scartata: nota in inglese",
      m_bad_en["calib"] is None and "outside" in m_bad_en["cal_note"], m_bad_en["cal_note"])
try:
    core.parse_mca(b"1\n2\n3\n")
    check("file senza <<DATA>> (en) → errore", False)
except core.McaError as exc:
    check("file senza <<DATA>> (en) → messaggio inglese",
          "<<DATA>>" in str(exc) and "does not look like" in str(exc), str(exc))
te = core.lld_table(sim, 300.0, 1.0, 50.0).set_index("Element")
check("LLD: intestazioni inglesi",
      "LLD 3√B [ppm] (ESTIMATE)" in te.columns and "Line used" in te.columns,
      ", ".join(te.columns))
cfg_en, _ = core.export_config(P, cfg0, 2.0, 300.0)
check("export config: commenti in inglese", "exported from the XRF GUI" in cfg_en
      and "Poisson noise and MCA channels" in cfg_en)
i18n.set_lang("it")

ALLOWED_LITERALS = {
    "fisso", "casuale", "conteggi/canale", "ph/s/keV",
    "fluorescenza", "coerente", "incoerente",
    "parametrica", "collimatore", "manuale",
    "nessuno", "nessuna", "live time", "su un picco",
    "Italiano", "English", "it", "en",
}
ITALIAN_UI = re.compile(
    r"\b(Simula|Conteggi|Atmosfera|Campione|Rivelatore|Parametri|Acquisizione|"
    r"Etichette|Fluorescenza|illuminata|Composizione|Tensione|"
    r"Entra|Esci|errata|Normalizzazione|misurato|simulato|"
    r"Note fisiche|Aggiungi al confronto|Nuova realizzazione|"
    r"limite di rivelabilità|Area illuminata|Filtro aggiuntivo|"
    r"Password errata|Interfaccia per)\b",
    re.I,
)
app_src = (HERE / "app.py").read_text(encoding="utf-8")
tree = ast.parse(app_src)
hits = []
doc_node = None
if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
    doc_node = tree.body[0].value

class _Lit(ast.NodeVisitor):
    def visit_Constant(self, node):
        if not isinstance(node.value, str):
            return
        s = node.value
        if node is doc_node or s in ALLOWED_LITERALS:
            return
        if ITALIAN_UI.search(s):
            hits.append(f"L{node.lineno}: {s[:80]!r}")

_Lit().visit(tree)
check("app.py: nessun testo UI italiano fuori da i18n", not hits,
      "; ".join(hits[:6]))

# 11. AppTest: render in entrambe le lingue ────────────────────────────────
if AppTest is not None:
    def _app_text(at_):
        parts = []
        for attr in ("markdown", "title", "header", "subheader", "caption",
                     "error", "warning", "info", "success", "text"):
            for el in getattr(at_, attr, []):
                parts.append(str(getattr(el, "value", "") or ""))
        for attr in ("slider", "radio", "selectbox", "number_input", "toggle",
                     "button", "text_input", "multiselect", "select_slider",
                     "checkbox", "form_submit_button", "download_button", "tabs"):
            for el in getattr(at_, attr, []):
                lab = getattr(el, "label", None) or getattr(el, "value", None)
                if lab:
                    parts.append(str(lab))
        return "\n".join(parts)

    os.environ.pop("XRF_APP_PASSWORD", None)
    at = AppTest.from_file(str(HERE / "app.py"), default_timeout=300).run()
    txt_it = _app_text(at)
    ok_it = (not at.exception and "### ⚙️ Parametri" in txt_it and "Corrente [µA]" in txt_it
             and "Tube current" not in txt_it and "### ⚙️ Parameters" not in txt_it)
    lang_radios = [r for r in at.radio if getattr(r, "key", None) == "lang"]
    if lang_radios:
        lang_radios[0].set_value("en")
        at.run()
    txt_en = _app_text(at)
    ok_en = (not at.exception and len(at.metric) == 6
             and "Tube current" in txt_en and "### ⚙️ Parameters" in txt_en
             and "▶ Simulate" in txt_en
             and "### ⚙️ Parametri" not in txt_en and "Corrente [µA]" not in txt_en)
    check("app: render IT e EN senza eccezioni, testi scambiati",
          ok_it and ok_en and bool(lang_radios),
          f"radio lingua {bool(lang_radios)}, IT ok {ok_it}, EN ok {ok_en}; "
          f"eccezioni EN {at.exception}")

print("TUTTO OK" if not FAILS else f"FALLITI: {FAILS}")
sys.exit(0 if not FAILS else 1)
