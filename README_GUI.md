# XRF sim — GUI Streamlit

Interfaccia grafica per il simulatore a parametri fondamentali `xrf_model` (v3).
**Non modifica il pacchetto**: la config è costruita in memoria partendo da `configs/lab_xrf.yml` e
`run_one_grid` è chiamato così com'è.

## File
| File | Cosa contiene |
|------|---------------|
| `app.py` | interfaccia Streamlit |
| `xrf_gui_core.py` | funzioni di calcolo (senza Streamlit): config, simulazione, canali MCA, Poisson, LLD, `.mca` |
| `test_functional.py` | test: confronto con `run_one_grid`, LLD, residui, filtri, `.mca`, broadening, app e password |
| `requirements_gui.txt` | dipendenze aggiuntive |
| `.streamlit/config.toml` | tema scuro |
| `.streamlit/secrets.toml.example` | modello per la password |

## Installazione (nel clone del repo)
Copia tutti i file sopra nella **root del repo**, accanto a `src/` e `configs/`
(in Finder `.streamlit/` è nascosta: Cmd+Shift+. per vederla).

```bash
cd ~/Documents/xrf-sim
source .venv/bin/activate
pip install -r requirements_gui.txt
python test_functional.py          # deve finire con "TUTTO OK" (~1 minuto)
streamlit run app.py
```
Si apre `http://localhost:8501`. Tutto gira in locale; Ctrl+C nel terminale per chiudere.
`xrf_gui_core.py` aggiunge `src/` al percorso da solo: non serve `PYTHONPATH`.

Da aggiungere al `.gitignore` del repo:
```
.streamlit/secrets.toml
```

## Password (facoltativa)
L'app chiede una password solo se ne è impostata una; altrimenti parte libera (uso locale).
- **In locale:** `cp .streamlit/secrets.toml.example .streamlit/secrets.toml` e cambia `app_password`,
  oppure `export XRF_APP_PASSWORD="..."` prima di `streamlit run app.py`.
- **Streamlit Community Cloud:** nella pagina dell'app, *Settings → Secrets*, incolla
  `app_password = "..."` e salva. Il file `secrets.toml` non va mai messo su git.

Il confronto usa `hmac.compare_digest`; lo stato di accesso resta nella sessione del browser, con il pulsante «Esci» in sidebar.

## Cosa fa
- **Sidebar** (si applica con ▶ Simula): campione da `data/compositions` o `.yml` caricato (frazioni o wt%,
  normalizzati a somma 1); kV, µA (avviso oltre 4 W); target Ag; filtri di `tube.filters` del config (si possono
  togliere) più un filtro aggiuntivo; aria/vuoto; griglia n×n; area illuminata (config, da collimatore, manuale);
  geometria del config o parametrica θ/D; rivelatore (area, Be, Si di default 500 µm con il valore del config accanto).
- **Acquisizione** (immediata): live time, Poisson con seed fisso o casuale e pulsante «Nuova realizzazione»,
  peaking time e τ/Tpeak per il dead time stimato.
- **Spettro**: Plotly, log/lin, conteggi/canale o ph/s/keV, etichette righe, Ag K Rayleigh/Compton (solo sopra
  25.5 kV), componenti, zona Ar. Picchi integrati analiticamente sui canali di destinazione (nessuno scalino).
- **Spettro misurato `.mca`**: calibrazione da `<<CALIBRATION>>` (keV o eV, lineare o quadratica con ≥3 coppie,
  controllo di B), live time con avviso se manca, pannello con DP5 CONFIGURATION e DPP STATUS (TPEA, GAIN,
  fast/slow count → dead time misurato), simulazione sui canali del misurato, normalizzazione per live time o
  su un picco, residui (mis − λ)/√λ con λ = simulazione riportata sulla scala del misurato.
- **Righe & LLD**: tabella righe; P/B e LLD per elemento su Kα e Lα (sceglie la migliore), definizione 3√B o
  Currie (L_D ≈ 2.71 + 4.65√B). Tutto marcato STIMA.
- **Confronto run**, **Strumento** (geometria, efficienze, spettro del tubo, composizione), **Note fisiche**.
- **Download**: `.mca` (REAL_TIME = LIVE_TIME/(1−DT stimato)), `.csv`, `.xlsx`, config YAML
  (zip con la composizione se il campione è stato caricato; l'intestazione elenca le chiavi rimosse).

## Note tecniche
- Senza i file di `detector_response` (`.npz`) e `mca_calibration` (`.calib`) queste chiavi sono tolte dalla
  config in memoria (banner informativo). Se ci sono, vengono usate.
- La simulazione gira a 1 µA ed è scalata alla corrente (modello lineare nella corrente). La cache dipende dai
  parametri e dall'impronta (mtime + dimensione) di `lab_xrf.yml` e del file di composizione.
- Dead time: dalla scheda FAST SDD (Tpeak 9.6 µs → 18 kcps in uscita al 50 % di dead time), con modello
  paralizzabile τ ≈ 2.0 × Tpeak. Il peaking time di default (4.8 µs) è una STIMA: quello vero è `TPEA` nel `.mca`.
- Tempi: 2×2 ≈ 1–3 s; primo run per kV/filtro/target +~7 s (SpekPy, cache in `.cache/tube`); 20×20 ≈ 100 s.

## Limiti del modello
1. La riga Ar K dell'aria (2.96 keV) e lo scattering dell'aria non sono modellati.
2. Nessun modello di footprint: la griglia è l'area illuminata.
3. Target Ag: config 1.0 µm, scheda Amptek 0.75 ± 0.1 µm.
4. Senza matrice Geant4 picchi gaussiani (niente escape, coda, shelf); pile-up non modellato. Con la matrice,
   tabella righe e LLD restano in approssimazione gaussiana.
5. La griglia SpekPy parte da ~1.02 keV: mancano le righe sotto 1 keV (O, C); Na Kα è al limite, Mg Kα c'è
   (in aria fortemente assorbita). P/B, LLD e dead time sono stime.
