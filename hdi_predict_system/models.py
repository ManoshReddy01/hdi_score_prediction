# models.py
# One SQLAlchemy class per box in the provided ER diagram, with the exact
# same table/column names so the schema in this project matches the
# diagram 1:1. Relationships (db.relationship) mirror the diagram's
# connecting lines and cardinalities (1:N, 1:1) so the ORM can walk from
# one entity to another the same way the diagram shows.
from datetime import datetime

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy

# Single shared SQLAlchemy instance, initialised against the Flask app
# in app.py via db.init_app(app). Keeping it here (not in app.py) avoids
# circular imports between app.py and models.py.
db = SQLAlchemy()


class User(db.Model, UserMixin):
    """ERD: USER. UserMixin bolts on the is_authenticated/is_active/etc.
    properties Flask-Login expects, so a User row can log in directly."""

    __tablename__ = "user"

    user_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)  # unique: used as the login key
    # NOTE: the diagram doesn't show a password column - it's added here
    # because a real "USER creates / starts SESSION" flow needs one to
    # authenticate; storing only the hash (never the raw password) is
    # standard practice.
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), default="researcher")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # ERD "starts" (1:N) and "creates" (1:N) edges out of USER.
    sessions = db.relationship("Session", backref="user", lazy=True)
    inputs = db.relationship("HDIInputData", backref="user", lazy=True)

    def get_id(self):
        # Flask-Login needs a string id; SQLAlchemy's PK is an int.
        return str(self.user_id)


class Session(db.Model):
    """ERD: SESSION. One row per login, closed out on logout so the
    table doubles as a lightweight audit trail of who used the tool
    and when."""

    __tablename__ = "session"

    session_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.user_id"), nullable=False)
    login_time = db.Column(db.DateTime, default=datetime.utcnow)
    logout_time = db.Column(db.DateTime, nullable=True)  # null while the session is still open
    status = db.Column(db.String(20), default="active")  # "active" -> "closed" on logout


class Country(db.Model):
    """ERD: COUNTRY. Reference/lookup table - seeded once from the
    generated dataset so every prediction can be tied back to a real
    country rather than a free-text field (keeps HDI_INPUT_DATA clean)."""

    __tablename__ = "country"

    country_id = db.Column(db.Integer, primary_key=True)
    country_name = db.Column(db.String(255), nullable=False, unique=True)
    region = db.Column(db.String(100))
    population = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # ERD "originates from" (1:N) edge into HDI_INPUT_DATA.
    inputs = db.relationship("HDIInputData", backref="country", lazy=True)


class Dataset(db.Model):
    """ERD: DATASET. Describes the training data used to build a model,
    kept separate from ML_MODEL so one dataset could (in principle) be
    reused to train several model versions - matching the diagram's
    1:N "trains" edge."""

    __tablename__ = "dataset"

    dataset_id = db.Column(db.Integer, primary_key=True)
    dataset_name = db.Column(db.String(255), nullable=False)
    source = db.Column(db.String(255))
    total_rows = db.Column(db.Integer)
    total_columns = db.Column(db.Integer)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    # ERD "trains" (1:N) edge into ML_MODEL.
    models = db.relationship("MLModel", backref="dataset", lazy=True)


class HDIInputData(db.Model):
    """ERD: HDI_INPUT_DATA. The four raw indicators a user submits for a
    country - this is the X input to the ML model. Kept as its own table
    (rather than folded into HDI_PREDICTION) so the same input could, in
    principle, be scored by more than one model version."""

    __tablename__ = "hdi_input_data"

    input_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.user_id"), nullable=False)
    country_id = db.Column(db.Integer, db.ForeignKey("country.country_id"), nullable=False)

    # The four UNDP HDI dimensions, exactly as named in the diagram.
    life_expectancy = db.Column(db.Float, nullable=False)
    mean_years_schooling = db.Column(db.Float, nullable=False)
    expected_years_schooling = db.Column(db.Float, nullable=False)
    gni_per_capita = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # ERD "results in" (1:1) edge into HDI_PREDICTION.
    # uselist=False makes this a single object instead of a list,
    # matching the diagram's 1:1 cardinality.
    prediction = db.relationship("HDIPrediction", backref="input_data", uselist=False, lazy=True)


class MLModel(db.Model):
    """ERD: ML_MODEL. Metadata about a trained model version - what
    algorithm was used and how well it performed - so predictions can
    always be traced back to exactly which model produced them."""

    __tablename__ = "ml_model"

    model_id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("dataset.dataset_id"), nullable=True)
    model_name = db.Column(db.String(255), nullable=False)
    algorithm_used = db.Column(db.String(100))
    accuracy_score = db.Column(db.Float)  # category-agreement rate on the held-out test split
    r2_score = db.Column(db.Float)        # regression fit quality on the raw HDI score
    model_file_path = db.Column(db.String(255))  # where the joblib .pkl lives on disk
    trained_at = db.Column(db.DateTime, default=datetime.utcnow)

    # ERD "predicts" (1:N) edge into HDI_PREDICTION.
    predictions = db.relationship("HDIPrediction", backref="model", lazy=True)


class HDIPrediction(db.Model):
    """ERD: HDI_PREDICTION. The ML model's output for one HDI_INPUT_DATA
    row: a numeric score plus the human-readable tier it falls into."""

    __tablename__ = "hdi_prediction"

    prediction_id = db.Column(db.Integer, primary_key=True)
    input_id = db.Column(db.Integer, db.ForeignKey("hdi_input_data.input_id"), nullable=False)
    model_id = db.Column(db.Integer, db.ForeignKey("ml_model.model_id"), nullable=False)

    predicted_hdi_score = db.Column(db.Float, nullable=False)  # 0.000 - 1.000
    hdi_category = db.Column(db.String(50), nullable=False)    # Low / Medium / High / Very High
    prediction_time = db.Column(db.DateTime, default=datetime.utcnow)

    # ERD "generates" (1:N) edge into VISUALIZATION_REPORT - one
    # prediction can generate several charts (e.g. gauge + breakdown).
    reports = db.relationship("VisualizationReport", backref="prediction", lazy=True)


class VisualizationReport(db.Model):
    """ERD: VISUALIZATION_REPORT. Pointer to a generated chart image for
    a prediction, so the result page (and history) can re-display past
    charts without regenerating them."""

    __tablename__ = "visualization_report"

    report_id = db.Column(db.Integer, primary_key=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey("hdi_prediction.prediction_id"), nullable=False)
    graph_path = db.Column(db.String(255))   # relative path under /static, e.g. "reports/pred_3_gauge.png"
    report_type = db.Column(db.String(50))   # "hdi_gauge" | "dimension_breakdown"
    generated_at = db.Column(db.DateTime, default=datetime.utcnow)
