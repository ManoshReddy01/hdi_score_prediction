# HDI Insight — Human Development Index Predictor

A Flask web app that predicts a country's Human Development Index (HDI)
score and tier (Low / Medium / High / Very High) from four UNDP
indicators, built to match the provided ER diagram exactly.

## Why things are built the way they are

| Decision | Reason |
|---|---|
| SQLite via Flask-SQLAlchemy | Zero external setup — the entire schema + data lives in one `instance/hdi.db` file, matching a "clone and run" project. |
| Dataset + model generated on first run, not shipped as binaries | Keeps the zip small and makes the ML pipeline fully reproducible from `data/countries.py` and a fixed random seed. |
| RandomForestRegressor predicting the raw score, then thresholded into a tier | The four UNDP dimensions combine non-linearly (geometric mean) — a regressor learns that shape, and reusing the *official* UNDP cutoffs (0.800 / 0.700 / 0.550) for the tier keeps the category meaningful rather than arbitrary. |
| Charts rendered to PNG files referenced by `VISUALIZATION_REPORT.graph_path` | Matches the diagram's design: a report is a *pointer* to a graph, not the graph itself, so results can be redisplayed without recomputation. |
| Custom CSS (no Bootstrap) | A small, deliberate design system (navy / paper / amber / teal / rose, Fraunces + Inter + JetBrains Mono) rather than a generic component library look. |

## Schema → code map

Every table below is a class in `models.py`, named and columned exactly
as in the ER diagram:

- **USER** → registers/logs in → **SESSION** (`starts`, 1:N)
- **USER** → picks a **COUNTRY** and enters indicators → **HDI_INPUT_DATA** (`creates` / `originates from`, both 1:N)
- **HDI_INPUT_DATA** → scored by the active **ML_MODEL** → **HDI_PREDICTION** (`results in` 1:1, `predicts` 1:N)
- **DATASET** → used to fit an **ML_MODEL** (`trains`, 1:N)
- **HDI_PREDICTION** → rendered as one or more **VISUALIZATION_REPORT** rows (`generates`, 1:N)

## Project layout

```
hdi_predictor/
├── app.py                  # Flask routes — the whole request flow
├── models.py                # SQLAlchemy models = the ER diagram, in code
├── config.py                 # Paths & settings, kept out of app.py
├── requirements.txt
├── data/
│   ├── countries.py          # ~120 real country names/regions used to seed data
│   └── generate_dataset.py    # Builds hdi_dataset.csv using the real UNDP formula
├── ml/
│   ├── train_model.py         # Trains + evaluates the RandomForest model
│   └── visualize.py           # Builds the gauge + dimension-breakdown PNGs
├── templates/                 # Jinja2 pages (see inline HTML comments)
└── static/
    ├── css/style.css          # The whole design system, as CSS variables
    └── reports/                # Generated chart PNGs land here at runtime
```

## Running it

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000**. On first launch the app will
automatically:
1. Generate `data/hdi_dataset.csv` (synthetic, UNDP-methodology-based)
2. Train `ml/hdi_model.pkl` and print its accuracy/R² to `ml/metrics.json`
3. Seed the `DATASET`, `COUNTRY`, and `ML_MODEL` tables in `instance/hdi.db`

Register an account, then use **New Prediction** to pick a country and
run the model — you'll land on a result page with two generated charts
(a tier gauge and a dimension breakdown) plus the full audit trail of
which input, model, and dataset produced them.

## Retraining with real data

Replace the contents of `data/hdi_dataset.csv` with a real UNDP HDI
extract (same column names: `life_expectancy`, `mean_years_schooling`,
`expected_years_schooling`, `gni_per_capita`, `hdi_score`,
`hdi_category`) and re-run `python ml/train_model.py`, then delete
`instance/hdi.db` so the app re-seeds `ML_MODEL`/`DATASET` from the new
metrics on next launch.
