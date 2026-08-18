"""
================================================================================
AI Agent for Brain Stroke Prediction
Electronic Health Records and Clinical Decision Support
================================================================================

One trained model: FT-Transformer.

Pipeline
--------
Load data -> Validate -> EDA -> Split -> Preprocess -> Train FT-Transformer
-> Early stopping on validation -> Calibration -> Test evaluation -> SHAP
-> Risk engine -> predict_record() -> Web interface

How to run in Google Colab
--------------------------
    !pip install -q shap gradio
    %run stroke_ftt.py            # use %run so the figures appear inline
    predict_record({...})         # score a patient in a new cell
    launch_web()                  # open the web form

DISCLAIMER
----------
Academic research prototype. Not a medical diagnostic tool. It does not diagnose
stroke and must not replace professional medical evaluation.
================================================================================
"""

import json
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ==============================================================================
# 1. SETTINGS
# ==============================================================================

DATA_PATH = "healthcare-dataset-stroke-data.csv"   # change if the CSV is elsewhere
ARTIFACTS = Path("artifacts")

SEED = 42            # random seed, used everywhere for reproducibility
EPOCHS = 100         # maximum number of training epochs
BATCH_SIZE = 64      # samples per gradient update
LEARNING_RATE = 1e-3
PATIENCE = 10        # stop after this many epochs without validation improvement

TARGET = "stroke"
NUMERIC = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
CATEGORICAL = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
FEATURES = NUMERIC + CATEGORICAL

# Plain-English names used in the report and the web interface.
LABELS = {
    "age": "Age", "avg_glucose_level": "Average glucose level", "bmi": "BMI",
    "hypertension": "Hypertension", "heart_disease": "Heart disease",
    "gender": "Gender", "ever_married": "Ever married", "work_type": "Work type",
    "Residence_type": "Residence type", "smoking_status": "Smoking status",
}

# Valid ranges for the numeric inputs, checked before any prediction.
RANGES = {
    "age": (0, 120), "avg_glucose_level": (20, 600), "bmi": (5, 100),
    "hypertension": (0, 1), "heart_disease": (0, 1),
}

# Prototype risk bands. These are demonstration thresholds, not clinical cut-offs.
RISK_BANDS = [(0.30, "Low"), (0.60, "Moderate"), (1.01, "High")]

DISCLAIMER = (
    "This is an academic research prototype and not a medical diagnostic tool. "
    "The output is a model estimate from a public dataset and must not replace "
    "professional medical evaluation."
)

# Short factual notes shown alongside the top contributing features.
NOTES = {
    "age": "Age is the strongest non-modifiable risk factor in this dataset.",
    "hypertension": "Hypertension is the strongest modifiable risk factor.",
    "avg_glucose_level": "Elevated glucose is associated with vascular damage.",
    "heart_disease": "Cardiac disease contributes through the cardioembolic pathway.",
    "bmi": "BMI acts largely through hypertension and metabolic pathways.",
}

# A synthetic patient used for the demonstration. Not a real person.
DEMO_PATIENT = {
    "gender": "Male", "age": 72.0, "hypertension": 1, "heart_disease": 1,
    "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
    "avg_glucose_level": 205.4, "bmi": 32.1, "smoking_status": "formerly smoked",
}

# Reference values from published tabular-benchmark literature on this dataset.
# These models are NOT trained by this code. They are shown only to place the
# FT-Transformer result in context.
REFERENCE_MODELS = pd.DataFrame([
    ["Logistic Regression", "0.10 - 0.16", "0.78 - 0.85", "baseline, linear"],
    ["Random Forest",       "0.09 - 0.15", "0.76 - 0.83", "baseline, tree ensemble"],
    ["DNN (MLP)",           "0.10 - 0.17", "0.77 - 0.84", "deep, dense"],
    ["TabNet",              "0.09 - 0.16", "0.75 - 0.83", "deep, attentive"],
], columns=["Reference model (not trained here)", "PR-AUC range", "ROC-AUC range", "Type"])


def set_seed(seed=SEED):
    """Seed Python, NumPy and PyTorch so every run gives the same result."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# 2. LOAD AND VALIDATE
# ==============================================================================

def step(number, title):
    """Print a numbered stage header so the terminal output reads as a pipeline."""
    print("\n" + "=" * 62)
    print(f"STEP {number}: {title}")
    print("=" * 62)


def load_data(path=DATA_PATH):
    """Read the CSV, check the columns exist, and fix the column types."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download healthcare-dataset-stroke-data.csv from "
            "https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset"
        )
    df = pd.read_csv(path)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError("Dataset is missing column(s): " + ", ".join(missing))

    # 'id' is dropped here: it is a record number, not a clinical measurement.
    df = df[FEATURES + [TARGET]].copy()
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL:
        df[col] = df[col].astype(str)
    df[TARGET] = df[TARGET].astype(int)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("Missing values per column:")
    print(df.isna().sum().to_string())
    print("Class distribution:")
    print(df[TARGET].value_counts().sort_index().to_string())
    print(f"  (0 = No stroke, 1 = Stroke; "
          f"{df[TARGET].mean():.2%} of records are positive)")
    return df


# ==============================================================================
# 3. EDA  (two figures)
# ==============================================================================

def run_eda(df):
    """Two figures: the class imbalance, and how stroke rate rises with age."""
    ARTIFACTS.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))

    # Figure 1 - class distribution. This is why accuracy is not the main metric.
    counts = df[TARGET].value_counts().sort_index()
    axes[0].bar(["No stroke", "Stroke"], counts.values, color=["#4C72B0", "#C44E52"])
    for i, value in enumerate(counts.values):
        axes[0].text(i, value, f"{value}\n({value / len(df):.1%})",
                     ha="center", va="bottom", fontsize=9)
    axes[0].set_title("Stroke class distribution", fontsize=11)
    axes[0].set_ylabel("Records")
    axes[0].set_ylim(0, counts.max() * 1.2)

    # Figure 2 - stroke rate by age band. The clearest signal in the dataset.
    bands = pd.cut(df["age"], [0, 30, 45, 60, 75, 120],
                   labels=["<30", "30-45", "45-60", "60-75", "75+"])
    rate = df.groupby(bands, observed=True)[TARGET].mean() * 100
    axes[1].bar(rate.index.astype(str), rate.values, color="#C44E52")
    for i, value in enumerate(rate.values):
        axes[1].text(i, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    axes[1].set_title("Stroke rate by age band", fontsize=11)
    axes[1].set_ylabel("% with stroke")
    axes[1].set_xlabel("Age band")

    plt.tight_layout()
    plt.savefig(ARTIFACTS / "eda.png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.show()


# ==============================================================================
# 4. SPLIT AND PREPROCESS
# ==============================================================================

def split_data(df):
    """Stratified 70 / 15 / 15 split. Stratified so each split keeps the same
    stroke rate, which matters when only ~5% of records are positive."""
    X, y = df[FEATURES], df[TARGET].values
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15 / 0.85, stratify=y_temp, random_state=SEED)
    print(f"Split: train={len(X_train)}  validation={len(X_val)}  test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


class Preprocessor:
    """Fills missing numbers, scales them, and turns categories into integers.

    Everything is learned from the TRAINING split only. Validation and test data
    are only transformed, never fitted, so no information leaks between splits.
    """

    def fit(self, X):
        self.medians, self.means, self.stds, self.categories = {}, {}, {}, {}
        for col in NUMERIC:
            values = X[col]
            self.medians[col] = float(values.median())
            filled = values.fillna(self.medians[col])
            self.means[col] = float(filled.mean())
            self.stds[col] = float(filled.std()) or 1.0
        for col in CATEGORICAL:
            # index 0 is kept for any category not seen during training
            self.categories[col] = ["__unknown__"] + sorted(X[col].unique())
        return self

    def transform(self, X):
        """Return (scaled numbers, category codes) as arrays the model can use."""
        num = np.zeros((len(X), len(NUMERIC)), dtype=np.float32)
        for i, col in enumerate(NUMERIC):
            values = X[col].fillna(self.medians[col])
            num[:, i] = (values.to_numpy() - self.means[col]) / self.stds[col]

        cat = np.zeros((len(X), len(CATEGORICAL)), dtype=np.int64)
        for i, col in enumerate(CATEGORICAL):
            lookup = {value: j for j, value in enumerate(self.categories[col])}
            cat[:, i] = [lookup.get(str(v), 0) for v in X[col]]
        return num, cat

    @property
    def cat_sizes(self):
        return [len(self.categories[c]) for c in CATEGORICAL]


# ==============================================================================
# 5. FT-TRANSFORMER
# ==============================================================================

class FTTransformer(nn.Module):
    """FT-Transformer = Feature Tokenizer + Transformer.

    Step 1  Every feature becomes a token (a vector of length d_token):
            - each number gets its own learned weight and bias
            - each category gets a learned embedding looked up by its code
    Step 2  A learned [CLS] token is added at the front of the sequence.
    Step 3  Transformer encoder layers apply multi-head self-attention, so each
            feature's representation can depend on the other features.
    Step 4  The final [CLS] vector goes through a small head to one logit.
    """

    def __init__(self, n_numeric, cat_sizes, d_token=32, n_layers=2, n_heads=4,
                 dropout=0.1):
        super().__init__()
        # Step 1a - one linear embedding per numerical feature
        self.num_weight = nn.Parameter(torch.randn(n_numeric, d_token) * 0.02)
        self.num_bias = nn.Parameter(torch.zeros(n_numeric, d_token))

        # Step 1b - one embedding table holding every category of every column
        self.cat_embed = nn.Embedding(sum(cat_sizes), d_token)
        offsets = np.concatenate([[0], np.cumsum(cat_sizes)[:-1]])
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

        # Step 2 - the [CLS] token that will summarise the whole patient
        self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

        # Step 3 - the transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Step 4 - classification head producing a single logit
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, 1))

    def forward(self, x_num, x_cat):
        num_tokens = x_num.unsqueeze(-1) * self.num_weight + self.num_bias
        cat_tokens = self.cat_embed(x_cat + self.offsets)
        tokens = torch.cat([num_tokens, cat_tokens], dim=1)
        tokens = torch.cat([self.cls.expand(len(tokens), -1, -1), tokens], dim=1)
        encoded = self.encoder(tokens)
        return self.head(encoded[:, 0])          # [CLS] output -> one logit


def train_model(prep, X_train, y_train, X_val, y_val):
    """Train the FT-Transformer with early stopping on validation PR-AUC.

    Loss      : BCEWithLogitsLoss with pos_weight, so the ~5% stroke cases are
                weighted up and the model cannot succeed by predicting 'no' always.
                It takes raw logits (no sigmoid before it) for numerical stability.
    Optimizer : AdamW.
    Stopping  : keep the weights from the best validation PR-AUC epoch.
    """
    set_seed()
    num_train, cat_train = prep.transform(X_train)
    num_val, cat_val = prep.transform(X_val)

    model = FTTransformer(len(NUMERIC), prep.cat_sizes).to(DEVICE)

    # pos_weight = negatives / positives  (about 19 on this dataset)
    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    num_t = torch.tensor(num_train, device=DEVICE)
    cat_t = torch.tensor(cat_train, device=DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)

    print(f"\nTraining FT-Transformer on {DEVICE} "
          f"(epochs<={EPOCHS}, batch={BATCH_SIZE}, lr={LEARNING_RATE}, "
          f"pos_weight={pos_weight:.1f})")

    best_score, best_state, waited = -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(y_t), device=DEVICE)
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            logits = model(num_t[batch], cat_t[batch]).squeeze(-1)
            loss = loss_fn(logits, y_t[batch])
            loss.backward()
            optimizer.step()

        val_probs = predict_proba(model, prep, X_val)
        score = average_precision_score(y_val, val_probs)     # validation PR-AUC

        if score > best_score:                                # improved: save weights
            best_score, waited = score, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:                                                 # no improvement
            waited += 1
            if waited >= PATIENCE:
                print(f"  early stop at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs)")
                break
        if epoch % 10 == 0:
            print(f"  epoch {epoch:3d} | loss {loss.item():.4f} | val PR-AUC {score:.4f}")

    model.load_state_dict(best_state)                          # restore the best epoch
    model.eval()
    print(f"  best validation PR-AUC: {best_score:.4f}")
    return model


def predict_proba(model, prep, X):
    """Return P(stroke) for each row. No gradients needed here."""
    num, cat = prep.transform(X)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(num, device=DEVICE),
                       torch.tensor(cat, device=DEVICE)).squeeze(-1)
        return torch.sigmoid(logits).cpu().numpy()


# ==============================================================================
# 6. CALIBRATION AND EVALUATION
# ==============================================================================

def fit_calibrator(val_probs, y_val):
    """Platt scaling: a small logistic regression that maps the model's raw
    probability to a better-calibrated one. Fitted on VALIDATION data only,
    because using the test set here would leak information."""
    logit = np.log(np.clip(val_probs, 1e-6, 1 - 1e-6) /
                   (1 - np.clip(val_probs, 1e-6, 1 - 1e-6)))
    return LogisticRegression(max_iter=1000).fit(logit.reshape(-1, 1), y_val)


def calibrate(calibrator, probs):
    """Apply the calibrator and keep the result strictly inside (0, 1)."""
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    logit = np.log(probs / (1 - probs)).reshape(-1, 1)
    return np.clip(calibrator.predict_proba(logit)[:, 1], 0.001, 0.999)


def pick_threshold(y_val, val_probs):
    """Choose the probability cut-off that maximises F1 on VALIDATION data.

    0.5 would be arbitrary here because the classes are so imbalanced: after
    calibration almost every patient scores below 0.5, so a 0.5 cut-off would
    flag nobody. Pass CALIBRATED probabilities so the cut-off is on the same
    scale as the values it will later be compared against.
    """
    precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    return float(thresholds[int(np.nanargmax(f1[:-1]))])


def evaluate(y_true, probs, threshold):
    """All reported metrics. PR-AUC is the primary one for imbalanced data."""
    y_pred = (probs >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probs),
        "PR-AUC": average_precision_score(y_true, probs),
        "Brier": brier_score_loss(y_true, probs),
    }


# ==============================================================================
# 7. SHAP, RISK ENGINE, DECISION SUPPORT
# ==============================================================================

def explain(record_df, top_n=5):
    """SHAP contributions for ONE patient only, returning the top 5 features.

    KernelExplainer compares the patient against a small background sample of
    training patients and shares out the change in the prediction between the
    features. If SHAP is unavailable we substitute each feature with its
    training reference value and measure how far the probability moves.
    """
    def model_predict(matrix):
        frame = pd.DataFrame(matrix, columns=FEATURES)
        for col in NUMERIC:
            frame[col] = pd.to_numeric(frame[col])
        return predict_proba(MODEL, PREP, frame)

    if HAS_SHAP:
        try:
            explainer = shap.KernelExplainer(model_predict, BACKGROUND[FEATURES].values)
            values = explainer.shap_values(record_df[FEATURES].values,
                                           nsamples=100, silent=True)
            series = pd.Series(np.array(values).reshape(-1), index=FEATURES)
            return series.reindex(series.abs().sort_values(ascending=False).index)[:top_n]
        except Exception as exc:
            print(f"(SHAP unavailable: {exc} - using reference substitution)")

    base = float(predict_proba(MODEL, PREP, record_df)[0])
    scores = {}
    for col in FEATURES:
        altered = record_df.copy()
        altered[col] = (PREP.medians[col] if col in NUMERIC
                        else BACKGROUND[col].mode()[0])
        scores[col] = base - float(predict_proba(MODEL, PREP, altered)[0])
    series = pd.Series(scores)
    return series.reindex(series.abs().sort_values(ascending=False).index)[:top_n]


def risk_band(probability):
    """Map a probability to Low / Moderate / High. Prototype thresholds only."""
    for limit, label in RISK_BANDS:
        if probability < limit:
            return label
    return "High"


def validate_record(record):
    """Check every field is present and inside a sensible range."""
    errors = []
    for col in NUMERIC:
        if col not in record:
            errors.append(f"Missing: {LABELS[col]}")
            continue
        try:
            value = float(record[col])
        except (TypeError, ValueError):
            errors.append(f"{LABELS[col]} must be a number")
            continue
        low, high = RANGES[col]
        if not low <= value <= high:
            errors.append(f"{LABELS[col]} must be between {low} and {high}")
    errors += [f"Missing: {LABELS[c]}" for c in CATEGORICAL if c not in record]
    return errors


def decision_support(probability, band, contributions):
    """Deterministic summary text. It suggests assessment, never a diagnosis
    and never a treatment."""
    factors = ", ".join(LABELS[name] for name in contributions.index[:3])
    return (
        f"The model places this patient in the '{band}' band with a model-estimated "
        f"stroke risk of {probability:.2f}. The values contributing most to this "
        f"estimate were: {factors}. Appropriate clinical assessment of the relevant "
        f"vascular risk factors should be considered by a qualified clinician."
    )


def predict_record(record, show=True):
    """Score one patient.

    Flow: validate -> preprocess -> FT-Transformer -> calibration -> SHAP
          -> risk engine -> decision support
    """
    errors = validate_record(record)
    if errors:
        if show:
            print("Cannot process this record:")
            for error in errors:
                print("  -", error)
        return {"ok": False, "errors": errors}

    frame = pd.DataFrame([{c: record[c] for c in FEATURES}])
    for col in NUMERIC:
        frame[col] = pd.to_numeric(frame[col])
    for col in CATEGORICAL:
        frame[col] = frame[col].astype(str)

    raw = float(predict_proba(MODEL, PREP, frame)[0])
    probability = float(calibrate(CALIBRATOR, np.array([raw]))[0])
    band = risk_band(probability)
    contributions = explain(frame)
    summary = decision_support(probability, band, contributions)

    result = {"ok": True, "probability": probability, "risk_band": band,
              "top_features": contributions, "summary": summary,
              "disclaimer": DISCLAIMER}

    if show:
        print("\n" + "=" * 70)
        print("STROKE RISK ASSESSMENT")
        print("=" * 70)
        for key, value in record.items():
            print(f"  {LABELS.get(key, key):<24}: {value}")
        print(f"\n  Model-estimated risk    : {probability:.3f}  ({probability:.1%})")
        print(f"  Risk band               : {band}")
        print("\n  Top contributing features (SHAP):")
        for name, value in contributions.items():
            arrow = "raises" if value > 0 else "lowers"
            print(f"    {LABELS[name]:<24} {value:+.4f}  {arrow} the estimate")
        print(f"\n  Decision support: {summary}")
        print(f"\n  {DISCLAIMER}")
        print("=" * 70)
    return result


# ==============================================================================
# 8. WEB INTERFACE  (Gradio)
# ==============================================================================

def launch_web(share=True):
    """A simple form that calls predict_record() - the same model, calibration,
    SHAP and risk engine used everywhere else in this file."""
    import gradio as gr

    def on_predict(age, gender, hypertension, heart_disease, ever_married,
                   work_type, residence, glucose, bmi, smoking):
        record = {
            "age": age, "gender": gender,
            "hypertension": 1 if hypertension == "Yes" else 0,
            "heart_disease": 1 if heart_disease == "Yes" else 0,
            "ever_married": ever_married, "work_type": work_type,
            "Residence_type": residence, "avg_glucose_level": glucose,
            "bmi": bmi, "smoking_status": smoking,
        }
        result = predict_record(record, show=False)
        if not result["ok"]:
            return "### Please check the input\n" + "\n".join(
                f"- {e}" for e in result["errors"])

        colour = {"Low": "#15803d", "Moderate": "#b45309", "High": "#b91c1c"}[result["risk_band"]]
        lines = [
            f"## <span style='color:{colour}'>{result['probability']:.1%} &mdash; "
            f"{result['risk_band']} band</span>",
            "",
            "**Top 5 contributing features (SHAP)**",
            "",
            "| Feature | Contribution | Effect |",
            "|---|---|---|",
        ]
        for name, value in result["top_features"].items():
            effect = "raises" if value > 0 else "lowers"
            lines.append(f"| {LABELS[name]} | `{value:+.4f}` | {effect} |")
        lines += ["", "**Decision support**", "", result["summary"],
                  "", f"> {DISCLAIMER}"]
        return "\n".join(lines)

    with gr.Blocks(title="Stroke Risk Prediction") as demo:
        gr.Markdown("# AI Agent for Brain Stroke Prediction\n"
                    "FT-Transformer on EHR data. "
                    "**Academic research prototype - not a medical diagnostic tool.**")
        with gr.Row():
            with gr.Column():
                age = gr.Number(label="Age (years)", value=67)
                gender = gr.Dropdown(PREP.categories["gender"][1:], label="Gender",
                                     value="Male")
                hypertension = gr.Radio(["No", "Yes"], label="Hypertension", value="Yes")
                heart_disease = gr.Radio(["No", "Yes"], label="Heart disease", value="No")
                ever_married = gr.Dropdown(PREP.categories["ever_married"][1:],
                                           label="Ever married", value="Yes")
            with gr.Column():
                work_type = gr.Dropdown(PREP.categories["work_type"][1:],
                                        label="Work type", value="Private")
                residence = gr.Dropdown(PREP.categories["Residence_type"][1:],
                                        label="Residence type", value="Urban")
                glucose = gr.Number(label="Average glucose level (mg/dL)", value=185.0)
                bmi = gr.Number(label="BMI", value=30.5)
                smoking = gr.Dropdown(PREP.categories["smoking_status"][1:],
                                      label="Smoking status", value="formerly smoked")

        button = gr.Button("Predict Stroke Risk", variant="primary")
        output = gr.Markdown()
        button.click(on_predict,
                     [age, gender, hypertension, heart_disease, ever_married,
                      work_type, residence, glucose, bmi, smoking],
                     output)

    demo.launch(share=share)
    return demo


# ==============================================================================
# 9. MAIN
# ==============================================================================

def main(path=DATA_PATH):
    """Run the whole pipeline and leave MODEL, PREP, CALIBRATOR ready to use."""
    global MODEL, PREP, CALIBRATOR, BACKGROUND, THRESHOLD

    set_seed()
    print("=" * 62)
    print("AI AGENT FOR BRAIN STROKE PREDICTION - FT-Transformer")
    print("=" * 62)

    step(1, "Loading dataset")
    df = load_data(path)

    step(2, "Exploratory data analysis")
    run_eda(df)

    step(3, "Train/Validation/Test split (70/15/15, stratified)")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    step(4, "Preprocessing (median impute + standardise, fitted on TRAIN only)")
    PREP = Preprocessor().fit(X_train)
    print(f"Imputation medians and category codes learned from "
          f"{len(X_train)} training records.")
    print("Validation and test data are transformed with these values, never refitted,")
    print("so no information leaks from the held-out splits.")

    step(5, "Building & training FT-Transformer (PyTorch)")
    MODEL = train_model(PREP, X_train, y_train, X_val, y_val)

    step(6, "Calibration & threshold selection (VALIDATION set)")
    val_probs = predict_proba(MODEL, PREP, X_val)
    CALIBRATOR = fit_calibrator(val_probs, y_val)
    val_cal = calibrate(CALIBRATOR, val_probs)
    THRESHOLD = pick_threshold(y_val, val_cal)
    val_metrics = evaluate(y_val, val_cal, THRESHOLD)

    print(f"Platt calibration fitted on {len(y_val)} validation records.")
    print(f"Decision threshold chosen on validation: {THRESHOLD:.3f} "
          f"(maximising F1)\n")
    print(f"Validation Accuracy  : {val_metrics['Accuracy']:.2%}")
    print(f"Validation Precision : {val_metrics['Precision']:.2%}")
    print(f"Validation Recall    : {val_metrics['Recall']:.2%}")
    print(f"Validation F1-score  : {val_metrics['F1']:.2%}")
    print(f"Validation ROC-AUC   : {val_metrics['ROC-AUC']:.4f}")
    print(f"Validation PR-AUC    : {val_metrics['PR-AUC']:.4f}")

    step(7, "Evaluating on TEST set")
    # The test split is touched once, here, after every choice has been made.
    test_probs = calibrate(CALIBRATOR, predict_proba(MODEL, PREP, X_test))
    metrics = evaluate(y_test, test_probs, THRESHOLD)
    y_pred = (test_probs >= THRESHOLD).astype(int)

    print(f"Testing Accuracy  : {metrics['Accuracy']:.2%}")
    print(f"Precision (Stroke): {metrics['Precision']:.2%}")
    print(f"Recall (Stroke)   : {metrics['Recall']:.2%}")
    print(f"F1-score (Stroke) : {metrics['F1']:.2%}")
    print(f"ROC-AUC           : {metrics['ROC-AUC']:.4f}")
    print(f"PR-AUC            : {metrics['PR-AUC']:.4f}  <-- primary metric")
    print(f"Brier score       : {metrics['Brier']:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["No Stroke", "Stroke"],
                                digits=2, zero_division=0))

    print("READ THE ACCURACY CAREFULLY")
    print(f"Predicting 'no stroke' for every patient already scores "
          f"{1 - y_test.mean():.2%} accuracy")
    print("on this split while detecting zero stroke cases. Any accuracy target")
    print("below that number is met by a model that does nothing, so accuracy is")
    print("reported but is NOT the metric this project is judged on.")
    print(f"The PR-AUC no-skill baseline is the stroke rate ({y_test.mean():.3f}),")
    print("not 0.5, so PR-AUC must be read against that.")

    step(8, "Reference comparison (published values, NOT trained here)")
    print(REFERENCE_MODELS.to_string(index=False))
    print(f"\nThis code trains ONE model: FT-Transformer "
          f"(PR-AUC {metrics['PR-AUC']:.4f}, ROC-AUC {metrics['ROC-AUC']:.4f}).")
    print("The rows above are published reference ranges for context only.")

    ARTIFACTS.mkdir(exist_ok=True)
    torch.save(MODEL.state_dict(), ARTIFACTS / "ft_transformer.pt")
    (ARTIFACTS / "results.json").write_text(json.dumps(
        {"test": {k: round(v, 4) for k, v in metrics.items()},
         "validation": {k: round(v, 4) for k, v in val_metrics.items()},
         "threshold": round(THRESHOLD, 4), "seed": SEED}, indent=2))

    BACKGROUND = X_train.sample(min(50, len(X_train)), random_state=SEED)

    step(9, "Demonstration patient (SHAP + risk engine)")
    demo = predict_record(DEMO_PATIENT)

    contributions = demo["top_features"].sort_values()
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.barh([LABELS[c] for c in contributions.index], contributions.values,
            color=["#C44E52" if v > 0 else "#4C72B0" for v in contributions.values])
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlabel("SHAP contribution to the model estimate")
    ax.set_title("Top 5 contributing features - demonstration patient", fontsize=11)
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "shap_demo.png", dpi=120, bbox_inches="tight",
                facecolor="white")
    plt.show()

    print("\nReady. Next steps:")
    print("  predict_record({...})     score your own patient")
    print("  launch_web(share=False)   open the web form")
    return metrics


if __name__ == "__main__":
    main()
