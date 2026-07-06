# config.py
# Central place for every path/secret the app needs, so nothing is
# hard-coded inside app.py, models.py, or the ML scripts. Keeping this
# separate makes it trivial to point the app at a different database or
# storage location later (e.g. moving from SQLite to Postgres) without
# touching business logic.
import os

# Absolute path to the project root, computed once here so every other
# path in this file (and in app.py) is reliable regardless of the
# working directory the app happens to be launched from.
BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    # Re-exposed as a class attribute so app.py can do Config.BASE_DIR
    # instead of importing the module-level constant separately.
    BASE_DIR = BASE_DIR

    # Used by Flask to sign session cookies. Overridable via an env var
    # so a real deployment never ships the placeholder value.
    SECRET_KEY = os.environ.get("HDI_SECRET_KEY", "hdi-dev-secret-change-me")

    # SQLite is used because it needs zero external setup - the whole
    # app (schema + data) is a single portable .db file, which matches
    # the "everything in one zip" nature of this project.
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(BASE_DIR, "instance", "hdi.db")
    # Disabled because it adds overhead and we don't use Flask-SQLAlchemy's
    # event system anywhere in this app.
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Where the synthetic dataset and trained model artifacts live.
    # Kept as folders (not baked into the DB) so they can be regenerated
    # or swapped out independently of the web app.
    DATA_DIR = os.path.join(BASE_DIR, "data")
    ML_DIR = os.path.join(BASE_DIR, "ml")
    DATASET_PATH = os.path.join(DATA_DIR, "hdi_dataset.csv")
    MODEL_PATH = os.path.join(ML_DIR, "hdi_model.pkl")

    # Every VISUALIZATION_REPORT.graph_path points into this folder, so
    # the generated PNGs are served directly by Flask's static handler
    # instead of needing a dedicated download route.
    REPORTS_DIR = os.path.join(BASE_DIR, "static", "reports")
