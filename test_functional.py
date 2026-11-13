"""Test funzionale: carica le funzioni di app.py (senza UI) e confronta con run_one_grid.
Uso: python test_functional.py   (dalla root del repo, con app.py accanto a src/)"""
import copy, json, logging, sys
from pathlib import Path
logging.disable(logging.WARNING)
src = Path("app.py").read_text(encoding="utf-8")
ns = {"__file__": str(Path("app.py").resolve()), "__name__": "app_funcs"}
exec(compile(src.split("# Header")[0], "app.py", "exec"), ns)
import numpy as np
from xrf_model.pipeline import run_one_grid
import contextlib, io, tempfile

cfg0 = ns["base_config"]()
g = ns["geometry_defaults"](cfg0)
P = dict(sample="soil_14163", custom_comp=None, secondary=True, kv=50, target_um=1.0, tube_be_mm=0.125,
         filter_mat="nessuno", filter_mm=0.0, theta_src=g["theta_src"], D_src=g["D_src"],
         theta_det=g["theta_det"], D_det=g["D_det"], det_area_mm2=17.0, atmosphere="air", src_vac=0.0,
         det_vac=0.0, grid_hx=0.4, grid_hy=0.4, grid_n=2, be_um=12.7, si_um=450.0, fano=0.118, w_ev=3.65,
         noise_ev=12.7, use_response=False)
ok = True
for atm in ("air", "vacuum"):
    p = dict(P, atmosphere=atm)
    sim = ns["scaled"](ns["simulate"](json.dumps(p, sort_keys=True)), 2.0)        # 2 µA = corrente di lab_xrf.yml
    cfg = copy.deepcopy(cfg0); cfg["atmosphere"] = atm
    for k in ("detector_response", "mca_calibration", "mca_channels"): cfg.pop(k, None)
    cfg["paths"]["output_dir"] = tempfile.mkdtemp()
    with contextlib.redirect_stdout(io.StringIO()):
        ref = run_one_grid(None, cfg)["soil_14163"]
    d1 = sim["fluo"].set_index(["Element", "Line"])["Intensity_ph_s"]
    d2 = ref["fluorescence"].set_index(["Element", "Line"])["Intensity_ph_s"]
    rel_lines = float(np.max(np.abs(d1 - d2) / np.maximum(d2, 1e-30)))
    rel_spec = float(np.max(np.abs(sim["spec"] - ref["spectrum"])) / np.max(ref["spectrum"]))
    gauss_sum = sim["comp_fluo"] + sim["comp_coh"] + sim["comp_inc"]
    rel_comp = float(np.max(np.abs(gauss_sum - ref["spectrum"])) / np.max(ref["spectrum"]))
    fe = d1[("Fe", "Kα1")]; si = d1[("Si", "Kα1")]
    print(f"[{atm}] Fe Kα1 {fe:.2f} ph/s  Si Kα1 {si:.3f} ph/s | righe vs pipeline {rel_lines:.1e} | "
          f"spettro {rel_spec:.1e} | somma componenti {rel_comp:.1e}")
    ok &= rel_lines < 1e-9 and rel_spec < 1e-9 and rel_comp < 1e-9
    lt = ns["lld_table"](sim, 300.0, 1.0, 50.0)
    print(lt[lt["Elemento"].isin(["Fe", "Rb", "Sr", "Y", "Zr", "Ca", "K"])][["Elemento", "Riga", "P/B", "LLD 3√B [ppm] (STIMA)"]].to_string(index=False))

# Poisson: media e varianza
lam = np.full(200000, 37.0)
x = np.random.default_rng(1).poisson(lam)
print(f"Poisson: media {x.mean():.2f} var {x.var():.2f} (attese 37)")
ok &= abs(x.mean() - 37) < 0.2 and abs(x.var() - 37) < 0.6

# .mca: scrittura e rilettura
counts = np.arange(100)
raw = ns["write_mca"](counts, (0.01, 0.02, 0.0), 300.0, 310.0, "test")
m = ns["parse_mca"](raw)
print("MCA roundtrip:", m["calib"], m["live"], m["real"], int(m["counts"].sum()))
ok &= m["counts"].sum() == counts.sum() and abs(m["calib"][1] - 0.02) < 1e-12 and m["live"] == 300.0
# formato Amptek con coppie canale-energia
amptek = b"<<PMCA SPECTRUM>>\nLIVE_TIME - 120.5\nREAL_TIME - 130\n<<CALIBRATION>>\nLABEL - keV\n100 0.6\n1000 6.0\n<<DATA>>\n1\n2\n3\n<<END>>\n<<DP5 CONFIGURATION>>\nGAIN=7.5;\n"
m = ns["parse_mca"](amptek)
print("Amptek pairs:", m["calib"], m["live"], m["counts"])
ok &= abs(m["calib"][1] - 0.006) < 1e-12 and abs(m["calib"][0]) < 1e-9 and m["live"] == 120.5
print("TUTTO OK" if ok else "FALLITO")
sys.exit(0 if ok else 1)
