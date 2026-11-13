# XRF sim — GUI Streamlit

Interfaccia grafica per il simulatore a parametri fondamentali `xrf_model` (v3).
**Non modifica il pacchetto**: la config è costruita in memoria partendo da `configs/lab_xrf.yml` e
`run_one_grid` è chiamato così com'è.

Lingua: selettore **Italiano / English** in cima alla sidebar (e nella pagina della password).
Il default è Italiano; la scelta resta nella sessione del browser.

Graphical interface for the fundamental-parameter simulator `xrf_model` (v3).
**It does not modify the package**: the config is built in memory from `configs/lab_xrf.yml` and
`run_one_grid` is called as is.

Language: **Italiano / English** selector at the top of the sidebar (and on the password page).
The default is Italian; the choice is kept in the browser session.

## File
| File | IT | EN |
|------|----|----|
| `app.py` | interfaccia Streamlit | Streamlit interface |
| `i18n.py` | testi IT/EN e funzione `t()` | IT/EN strings and `t()` helper |
| `xrf_gui_core.py` | funzioni di calcolo (senza Streamlit): config, simulazione, canali MCA, Poisson, LLD, `.mca` | calculation functions (no Streamlit): config, simulation, MCA channels, Poisson, LLD, `.mca` |
| `test_functional.py` | test: confronto con `run_one_grid`, LLD, residui, filtri, `.mca`, broadening, app, password e lingue | tests: comparison with `run_one_grid`, LLD, residuals, filters, `.mca`, broadening, app, password and languages |
| `requirements_gui.txt` | dipendenze aggiuntive | extra dependencies |
| `.streamlit/config.toml` | tema scuro | dark theme |
| `.streamlit/secrets.toml.example` | modello per la password | password template |

## Installazione / Installation (nel clone del repo / in the repo clone)
Copia tutti i file sopra nella **root del repo**, accanto a `src/` e `configs/`
(in Finder `.streamlit/` è nascosta: Cmd+Shift+. per vederla).

Copy all of the files above into the **repo root**, next to `src/` and `configs/`
(on macOS Finder `.streamlit/` is hidden: Cmd+Shift+. to show it).

```bash
cd ~/Documents/xrf-sim
source .venv/bin/activate
pip install -r requirements_gui.txt
python test_functional.py          # deve finire con "TUTTO OK" / must end with "TUTTO OK" (~1–2 min)
streamlit run app.py
```
Si apre `http://localhost:8501`. Tutto gira in locale; Ctrl+C nel terminale per chiudere.
`xrf_gui_core.py` aggiunge `src/` al percorso da solo: non serve `PYTHONPATH`.

It opens `http://localhost:8501`. Everything runs locally; Ctrl+C in the terminal to quit.
`xrf_gui_core.py` adds `src/` to the path itself: `PYTHONPATH` is not needed.

Da aggiungere al `.gitignore` del repo / add this to the repo `.gitignore`:
```
.streamlit/secrets.toml
```

## Password (facoltativa / optional)
L'app chiede una password solo se ne è impostata una; altrimenti parte libera (uso locale).
The app prompts for a password only if one is set; otherwise it starts unlocked (local use).

- **In locale / locally:** `cp .streamlit/secrets.toml.example .streamlit/secrets.toml` e cambia `app_password`,
  oppure / or `export XRF_APP_PASSWORD="..."` prima di / before `streamlit run app.py`.
- **Streamlit Community Cloud:** nella pagina dell'app, *Settings → Secrets*, incolla
  `app_password = "..."` e salva. Il file `secrets.toml` non va mai messo su git.
  On the app page, *Settings → Secrets*, paste `app_password = "..."` and save. Never commit `secrets.toml`.

Il confronto usa `hmac.compare_digest`; lo stato di accesso resta nella sessione del browser, con il pulsante «Esci» / «Sign out» in sidebar.

The comparison uses `hmac.compare_digest`; the signed-in state stays in the browser session, with the «Esci» / «Sign out» button in the sidebar.

## Cosa fa / What it does
- **Sidebar** (si applica con ▶ Simula / applied with ▶ Simulate): campione da `data/compositions` o `.yml` caricato (frazioni o wt%,
  normalizzati a somma 1); kV, µA (avviso oltre 4 W); target Ag; filtri di `tube.filters` del config (si possono
  togliere) più un filtro aggiuntivo; aria/vuoto; griglia n×n; area illuminata (config, da collimatore, manuale);
  geometria del config o parametrica θ/D; rivelatore (area, Be, Si di default 500 µm con il valore del config accanto).
  Sample from `data/compositions` or an uploaded `.yml` (fractions or wt%, normalized to sum 1); kV, tube current in µA
  (warning above 4 W); Ag target; config `tube.filters` (can be removed) plus an extra filter; air/vacuum; n×n grid;
  illuminated area (config, from collimator, manual); config or parametric θ/D geometry; detector (area, Be, Si default 500 µm,
  with the config value shown beside it).
- **Acquisizione / Acquisition** (immediata / immediate): live time, Poisson with a fixed or random seed and a «Nuova realizzazione» / «New realization» button,
  peaking time and τ/Tpeak for the estimated dead time.
- **Spettro / Spectrum**: Plotly, log/lin, counts/channel or ph/s/keV, line labels, Ag K Rayleigh/Compton (only above
  25.5 kV), components, Ar region. Peaks integrated analytically onto the destination channels (no steps).
- **Spettro misurato / Measured spectrum `.mca`**: calibration from `<<CALIBRATION>>` (keV or eV, linear or quadratic with ≥3 pairs,
  check on B), live time with a warning if missing, panel with DP5 CONFIGURATION and DPP STATUS (TPEA, GAIN,
  fast/slow count → measured dead time), simulation on the measured channels, normalization by live time or
  to a peak, residuals (meas − λ)/√λ with λ = simulation reported on the measured scale.
- **Righe & LLD / Lines & LLD**: line table; P/B and LLD per element on Kα and Lα (the better one is chosen), 3√B or
  Currie definition (L_D ≈ 2.71 + 4.65√B). Everything marked STIMA / ESTIMATE.
- **Confronto run / Run comparison**, **Strumento / Instrument** (geometry, efficiencies, tube spectrum, composition), **Note fisiche / Physical notes**.
- **Download**: `.mca` (REAL_TIME = LIVE_TIME/(1− estimated DT)), `.csv`, `.xlsx`, config YAML
  (zip with the composition if the sample was uploaded; the header lists the removed keys).

## Note tecniche / Technical notes
- Senza i file di `detector_response` (`.npz`) e `mca_calibration` (`.calib`) queste chiavi sono tolte dalla
  config in memoria (banner informativo). Se ci sono, vengono usate.
  Without the `detector_response` (`.npz`) and `mca_calibration` (`.calib`) files these keys are dropped from the
  in-memory config (info banner). If they are present, they are used.
- La simulazione gira a 1 µA ed è scalata alla corrente (modello lineare nella corrente). La cache dipende dai
  parametri e dall'impronta (mtime + dimensione) di `lab_xrf.yml` e del file di composizione.
  The simulation runs at 1 µA and is scaled to the tube current (model linear in current). The cache depends on the
  parameters and on the fingerprint (mtime + size) of `lab_xrf.yml` and of the composition file.
- Dead time: dalla scheda FAST SDD (Tpeak 9.6 µs → 18 kcps in uscita al 50 % di dead time), con modello
  paralizzabile / paralyzable τ ≈ 2.0 × Tpeak. Il peaking time di default (4.8 µs) è una STIMA / ESTIMATE: quello vero è `TPEA` nel `.mca`.
- Tempi / timings: 2×2 ≈ 1–3 s; primo run per kV/filtro/target +~7 s (SpekPy, cache in `.cache/tube`); 20×20 ≈ 100 s.

## Limiti del modello / Model limitations
1. La riga Ar K dell'aria (2.96 keV) e lo scattering dell'aria non sono modellati.
   The air Ar K line (2.96 keV) and air scattering are not modelled.
2. Nessun modello di footprint: la griglia è l'area illuminata.
   No footprint model: the grid is the illuminated area.
3. Target Ag: config 1.0 µm, scheda Amptek 0.75 ± 0.1 µm.
4. Senza matrice Geant4 picchi gaussiani (niente escape, coda, shelf); pile-up non modellato. Con la matrice,
   tabella righe e LLD restano in approssimazione gaussiana.
   Without the Geant4 matrix, peaks are Gaussians (no escape, tail or shelf); pile-up is not modelled. With the matrix,
   the line table and the LLD remain in the Gaussian approximation.
5. La griglia SpekPy parte da ~1.02 keV: mancano le righe sotto 1 keV (O, C); Na Kα è al limite, Mg Kα c'è
   (in aria fortemente assorbita). P/B, LLD e dead time sono stime.
   The SpekPy grid starts at ~1.02 keV: lines below 1 keV (O, C) are missing; Na Kα is at the edge, Mg Kα is present
   (strongly absorbed in air). P/B, LLD and dead time are estimates.
