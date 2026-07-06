# app.py
# The whole web app in one file (kept flat, rather than split into
# Blueprints, because the project is small enough that indirection would
# cost more than it saves). The request flow this file implements is:
#
#   1. USER registers/logs in           -> creates USER / SESSION rows
#   2. USER opens "New Prediction"      -> picks a COUNTRY, enters indicators
#   3. Form submit                      -> writes an HDI_INPUT_DATA row
#   4. The trained ML_MODEL scores it   -> writes an HDI_PREDICTION row
#   5. Two charts are rendered to disk  -> writes VISUALIZATION_REPORT rows
#   6. Result page shows the charts + the underlying numbers
#
# Every one of those nouns is a table defined in models.py, so this file
# is really just "wire the ERD up to HTTP routes."
import os
from datetime import datetime

from flask import Flask, flash, redirect, render_template, request, url_for
from flask_login import (
    LoginManager,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from werkzeug.security import check_password_hash, generate_password_hash

from config import Config
from data.generate_dataset import build as build_dataset
from ml.train_model import train as train_model
from ml.visualize import generate_dimension_chart, generate_gauge_chart
from models import (
    Country,
    Dataset,
    HDIInputData,
    HDIPrediction,
    MLModel,
    Session as SessionModel,
    User,
    VisualizationReport,
    db,
)

import joblib

app = Flask(__name__)
app.config.from_object(Config)

os.makedirs(os.path.join(Config.BASE_DIR, "instance"), exist_ok=True)
os.makedirs(Config.REPORTS_DIR, exist_ok=True)

db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.login_message = "Please sign in to continue."
login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ---------------------------------------------------------------------------
# Bootstrap: dataset + model + reference data, created once on first launch
# ---------------------------------------------------------------------------
def bootstrap():
    # Creates every table from models.py if the SQLite file is empty.
    # Safe to call on every startup - SQLAlchemy skips tables that
    # already exist.
    db.create_all()

    # The dataset CSV is generated on first run (not committed as a
    # static asset) so the whole project stays small and the data is
    # reproducibly derived from data/countries.py + the fixed RNG seed.
    if not os.path.exists(Config.DATASET_PATH):
        build_dataset(Config.DATASET_PATH)

    # Likewise, the model is trained on first run rather than shipped
    # as a binary - this keeps the zip lightweight and makes it obvious
    # exactly how the model was produced (see ml/train_model.py).
    if not os.path.exists(Config.MODEL_PATH):
        train_model(Config.DATASET_PATH, Config.MODEL_PATH, os.path.join(Config.ML_DIR, "metrics.json"))

    # Populate the DATASET + COUNTRY reference tables exactly once, so
    # re-running the app doesn't duplicate 120 country rows every time.
    if Dataset.query.count() == 0:
        import pandas as pd

        df = pd.read_csv(Config.DATASET_PATH)
        dataset_row = Dataset(
            dataset_name="Global HDI Indicators (synthetic, UNDP-methodology)",
            source="Generated via data/generate_dataset.py",
            total_rows=len(df),
            total_columns=len(df.columns),
        )
        db.session.add(dataset_row)
        db.session.commit()

        for _, row in df.iterrows():
            if not Country.query.filter_by(country_name=row["country_name"]).first():
                db.session.add(
                    Country(
                        country_name=row["country_name"],
                        region=row["region"],
                        population=int(row["population"]),
                    )
                )
        db.session.commit()

    # Record the trained model as an ML_MODEL row so every future
    # prediction can point at model_id and the UI can show the
    # accuracy/R^2 the model actually achieved, instead of a made-up number.
    if MLModel.query.count() == 0:
        import json

        metrics_path = os.path.join(Config.ML_DIR, "metrics.json")
        metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
        dataset_row = Dataset.query.first()
        db.session.add(
            MLModel(
                dataset_id=dataset_row.dataset_id if dataset_row else None,
                model_name="HDI RandomForest v1",
                algorithm_used=metrics.get("algorithm_used", "RandomForestRegressor"),
                accuracy_score=metrics.get("accuracy_score", 0.0),
                r2_score=metrics.get("r2_score", 0.0),
                model_file_path=Config.MODEL_PATH,
            )
        )
        db.session.commit()


# Cached in a module-level variable so the (relatively expensive)
# joblib.load only happens once per process, not on every /predict
# request.
_MODEL_BUNDLE = None


def get_model_bundle():
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        _MODEL_BUNDLE = joblib.load(Config.MODEL_PATH)
    return _MODEL_BUNDLE


def category_from_score(score):
    # Official UNDP HDI tier boundaries - used both when labelling the
    # training data (see data/generate_dataset.py) and here when
    # labelling a live prediction, so the two stay consistent.
    if score >= 0.800:
        return "Very High"
    if score >= 0.700:
        return "High"
    if score >= 0.550:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Public pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    stats = {
        "countries": Country.query.count(),
        "predictions": HDIPrediction.query.count(),
        "users": User.query.count(),
    }
    return render_template("index.html", stats=stats)


# --- register: creates a USER row only. No SESSION yet, since signing
# up is not the same event as logging in (matches the diagram: USER
# "starts" SESSION is a separate edge from USER itself). --------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        role = request.form.get("role", "researcher")

        if not name or not email or not password:
            flash("All fields are required.", "error")
        elif User.query.filter_by(email=email).first():
            flash("An account with that email already exists.", "error")
        else:
            user = User(
                name=name,
                email=email,
                password_hash=generate_password_hash(password),
                role=role,
            )
            db.session.add(user)
            db.session.commit()
            flash("Account created. Please sign in.", "success")
            return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password_hash, password):
            login_user(user)  # Flask-Login: marks this browser session as authenticated
            # ERD "starts" edge: a successful login is exactly the event
            # that creates a new SESSION row.
            sess = SessionModel(user_id=user.user_id, status="active")
            db.session.add(sess)
            db.session.commit()
            flash(f"Welcome back, {user.name}.", "success")
            return redirect(url_for("dashboard"))

        flash("Incorrect email or password.", "error")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    # Find this user's most recent open SESSION row and close it out,
    # so SESSION.status/logout_time reflect reality instead of every
    # session looking permanently "active".
    open_session = (
        SessionModel.query.filter_by(user_id=current_user.user_id, status="active")
        .order_by(SessionModel.session_id.desc())
        .first()
    )
    if open_session:
        open_session.status = "closed"
        open_session.logout_time = datetime.utcnow()
        db.session.commit()
    logout_user()
    flash("You have been signed out.", "success")
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# Authenticated app
# ---------------------------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    my_predictions = (
        HDIPrediction.query.join(HDIInputData)
        .filter(HDIInputData.user_id == current_user.user_id)
        .order_by(HDIPrediction.prediction_time.desc())
        .limit(6)
        .all()
    )
    model = MLModel.query.order_by(MLModel.trained_at.desc()).first()
    dataset = Dataset.query.first()
    return render_template(
        "dashboard.html",
        predictions=my_predictions,
        model=model,
        dataset=dataset,
        country_count=Country.query.count(),
    )


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    countries = Country.query.order_by(Country.country_name).all()

    # Build a country_id -> {indicator: value} lookup straight from the
    # generator's CSV. This is ONLY used client-side to auto-fill the form
    # with plausible starting numbers when a country is picked, so a user
    # testing the tool isn't stuck guessing GNI figures from scratch.
    # It never touches the database and is not used for the prediction itself
    # (the prediction always uses whatever values are actually submitted).
    import json as _json

    import pandas as pd

    seed_map = {}
    try:
        df = pd.read_csv(Config.DATASET_PATH)
        for c in countries:
            match = df[df["country_name"] == c.country_name]
            if not match.empty:
                r = match.iloc[0]
                seed_map[c.country_id] = {
                    "life_expectancy": float(r["life_expectancy"]),
                    "mean_years_schooling": float(r["mean_years_schooling"]),
                    "expected_years_schooling": float(r["expected_years_schooling"]),
                    "gni_per_capita": float(r["gni_per_capita"]),
                }
    except FileNotFoundError:
        pass  # form still works with manual entry if the CSV is ever missing
    seed_map_json = _json.dumps(seed_map)

    if request.method == "POST":
        try:
            country_id = int(request.form["country_id"])
            life_expectancy = float(request.form["life_expectancy"])
            mys = float(request.form["mean_years_schooling"])
            eys = float(request.form["expected_years_schooling"])
            gni = float(request.form["gni_per_capita"])
        except (KeyError, ValueError):
            flash("Please fill in every field with a valid number.", "error")
            return render_template("predict.html", countries=countries, seed_map_json=seed_map_json)

        country = Country.query.get(country_id)
        if not country:
            flash("Please choose a valid country.", "error")
            return render_template("predict.html", countries=countries, seed_map_json=seed_map_json)

        # Step 3 of the flow: persist exactly what the user submitted as
        # its own HDI_INPUT_DATA row, before any model touches it. This
        # keeps a permanent, un-modified record of the input.
        input_row = HDIInputData(
            user_id=current_user.user_id,
            country_id=country.country_id,
            life_expectancy=life_expectancy,
            mean_years_schooling=mys,
            expected_years_schooling=eys,
            gni_per_capita=gni,
        )
        db.session.add(input_row)
        db.session.commit()  # commit first so input_row.input_id is assigned

        # Step 4: run the trained RandomForest on the four raw indicators.
        # Feature order here MUST match FEATURES in ml/train_model.py.
        bundle = get_model_bundle()
        features = [[life_expectancy, mys, eys, gni]]
        score = float(bundle["model"].predict(features)[0])
        score = max(0.001, min(0.999, score))  # HDI is defined on (0, 1]; clamp defensively
        cat = category_from_score(score)

        # Always attribute the prediction to whichever model is
        # currently newest, so re-training later automatically applies
        # to new predictions without any other code change.
        model_row = MLModel.query.order_by(MLModel.trained_at.desc()).first()

        pred = HDIPrediction(
            input_id=input_row.input_id,
            model_id=model_row.model_id,
            predicted_hdi_score=round(score, 4),
            hdi_category=cat,
        )
        db.session.add(pred)
        db.session.commit()  # commit first so pred.prediction_id is assigned

        # Step 5: render the two charts to disk. Filenames are keyed by
        # prediction_id so every prediction gets its own, never-overwritten
        # image pair.
        gauge_path = os.path.join(Config.REPORTS_DIR, f"pred_{pred.prediction_id}_gauge.png")
        dim_path = os.path.join(Config.REPORTS_DIR, f"pred_{pred.prediction_id}_dimensions.png")
        generate_gauge_chart(score, cat, gauge_path)
        generate_dimension_chart(life_expectancy, mys, eys, gni, dim_path)

        # One HDI_PREDICTION -> many VISUALIZATION_REPORT rows (1:N per
        # the diagram) - here, one row per chart type generated above.
        db.session.add_all(
            [
                VisualizationReport(
                    prediction_id=pred.prediction_id,
                    graph_path=f"reports/pred_{pred.prediction_id}_gauge.png",
                    report_type="hdi_gauge",
                ),
                VisualizationReport(
                    prediction_id=pred.prediction_id,
                    graph_path=f"reports/pred_{pred.prediction_id}_dimensions.png",
                    report_type="dimension_breakdown",
                ),
            ]
        )
        db.session.commit()

        return redirect(url_for("result", prediction_id=pred.prediction_id))

    return render_template("predict.html", countries=countries, seed_map_json=seed_map_json)


@app.route("/result/<int:prediction_id>")
@login_required
def result(prediction_id):
    pred = HDIPrediction.query.get_or_404(prediction_id)
    input_row = pred.input_data
    if input_row.user_id != current_user.user_id:
        flash("You can only view your own predictions.", "error")
        return redirect(url_for("dashboard"))

    country = Country.query.get(input_row.country_id)
    reports = VisualizationReport.query.filter_by(prediction_id=pred.prediction_id).all()
    return render_template("result.html", pred=pred, input_row=input_row, country=country, reports=reports)


@app.route("/history")
@login_required
def history():
    rows = (
        db.session.query(HDIPrediction, HDIInputData, Country)
        .join(HDIInputData, HDIPrediction.input_id == HDIInputData.input_id)
        .join(Country, HDIInputData.country_id == Country.country_id)
        .filter(HDIInputData.user_id == current_user.user_id)
        .order_by(HDIPrediction.prediction_time.desc())
        .all()
    )
    return render_template("history.html", rows=rows)


@app.route("/countries")
@login_required
def countries():
    q = request.args.get("region", "")
    query = Country.query
    if q:
        query = query.filter_by(region=q)
    all_countries = query.order_by(Country.country_name).all()
    regions = sorted({c.region for c in Country.query.all() if c.region})
    return render_template("countries.html", countries=all_countries, regions=regions, selected_region=q)


@app.route("/model-info")
@login_required
def model_info():
    model = MLModel.query.order_by(MLModel.trained_at.desc()).first()
    dataset = Dataset.query.first()
    total_predictions = HDIPrediction.query.count()
    return render_template("model_info.html", model=model, dataset=dataset, total_predictions=total_predictions)


with app.app_context():
    bootstrap()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
