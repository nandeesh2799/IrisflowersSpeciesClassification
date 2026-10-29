# 🌸 Iris Flower Species Classification (Modular Engine & Web UI)

[![CI](https://github.com/nandeesh2799/IrisflowersSpeciesClassification/actions/workflows/ci.yml/badge.svg)](https://github.com/nandeesh2799/IrisflowersSpeciesClassification/actions/workflows/ci.yml)
[![Tests Passing](https://img.shields.io/badge/Unit%20Tests-32%2F32%20Passing-brightgreen?style=flat&logo=pytest)](https://github.com/nandeesh2799/IrisflowersSpeciesClassification)
[![Coverage](https://img.shields.io/badge/Coverage-92%25-success)](https://github.com/nandeesh2799/IrisflowersSpeciesClassification)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)](https://www.docker.com/)

An enterprise-ready, modular Machine Learning library, interactive command-line interface (CLI), and Streamlit web application for botanical species classification using the classic Iris dataset.

---

## 🏛️ System Architecture

```
IrisflowersSpeciesClassification/
├── src/
│   └── iris_classifier/            # Core installable Python package
│       ├── __init__.py
│       ├── cli.py                  # CLI command runner (iris-cli)
│       └── core/
│           ├── schemas.py          # Domain dataclasses & input validation
│           ├── data.py             # Data loading, cleaning, & stratified splitting
│           ├── trainer.py          # Cross-validation & grid search pipeline
│           └── predictor.py        # Self-bootstrapping inference engine
├── app/
│   ├── app.py                      # Glassmorphism Streamlit UI
│   └── utils.py                    # Backward-compatible bridge to iris_classifier
├── assets/images/                  # Botanical species photographic reference
├── data/                           # Canonical dataset (iris.csv)
├── models/                         # Serialized model (.pkl) and metadata (.json)
├── notebooks/                      # Exploratory data analysis & PPT generator
├── outputs/
│   ├── plots/                      # High-resolution pairplots & heatmaps
│   └── ppt/                        # Auto-generated presentation (.pptx)
├── tests/                          # Comprehensive pytest test suite (32 tests)
├── .github/workflows/ci.yml        # Multi-python matrix CI workflow
├── Dockerfile                      # Production container image
└── docker-compose.yml              # Container orchestration & healthchecks
```

---

## ⚡ Quickstart

### 1. Installation

Clone the repository and install the package with development dependencies:

```bash
git clone https://github.com/nandeesh2799/IrisflowersSpeciesClassification.git
cd IrisflowersSpeciesClassification

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package in editable mode
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 2. Command-Line Interface (`iris-cli`)

The package provides a first-class CLI for inference, evaluation, and training:

#### Species Prediction

```bash
# Predict in human-readable formatted text
iris-cli predict --sepal-length 5.1 --sepal-width 3.5 --petal-length 1.4 --petal-width 0.2

# Output structured JSON for automation & API pipelines
iris-cli predict --sepal-length 6.5 --sepal-width 3.0 --petal-length 5.5 --petal-width 1.8 --format json
```

Example JSON output:
```json
{
  "predicted_species": "virginica",
  "confidence": 0.7824,
  "probabilities": {
    "setosa": 0.0009,
    "versicolor": 0.2167,
    "virginica": 0.7824
  },
  "common_name": "Iris virginica",
  "description": "Iris virginica (Virginia iris) features the largest petal dimensions..."
}
```

#### Model Evaluation & Metrics

```bash
iris-cli evaluate
```

#### Retrain Model Pipeline

```bash
iris-cli train --random-state 42
```

---

## 🌐 Streamlit Web Application

Launch the interactive web application featuring glassmorphic UI, real-time probability distributions, and botanical photographic references:

```bash
streamlit run app/app.py
```

Open `http://localhost:8501` in your browser.

> **Resilient Self-Bootstrapping**: If the model artifact (`models/best_model.pkl`) is ever deleted or missing, the application automatically triggers the training pipeline and self-heals without crashing.

---

## 🐳 Docker Deployment

Run the complete application via Docker or Docker Compose:

### Using Docker Compose:

```bash
docker compose up -d --build
```

Access the web UI at `http://localhost:8501`. Check container health:

```bash
docker compose ps
```

### Using Docker CLI:

```bash
docker build -t iris-classifier:latest .
docker run -d -p 8501:8501 --name iris-app iris-classifier:latest
```

---

## 🧪 Testing & Quality Assurance

The codebase includes 32 automated unit tests validating input boundary invariants, model training convergence, self-bootstrapping fallbacks, and CLI behavior.

```bash
# Run pytest with code coverage report
pytest -v --cov=iris_classifier tests/
```

### CI/CD Workflow

Every commit and pull request runs automated tests against Python **3.10**, **3.11**, and **3.12** alongside container build validation via `.github/workflows/ci.yml`.

---

## 📊 Machine Learning Model Comparison

The pipeline trains and evaluates candidate models via `GridSearchCV` using 5-fold stratified cross-validation on holdout data:

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
|---|:---:|:---:|:---:|:---:|
| **Logistic Regression (Standardized)** | **93.3%** | **93.3%** | **93.3%** | **93.3%** |
| **Decision Tree Classifier** | 93.3% | 93.3% | 93.3% | 93.3% |
| **Random Forest Classifier** | 93.3% | 93.3% | 93.3% | 93.3% |

---

## 👨‍💻 Author

**Nandeesh N K**  
GitHub: [@nandeesh2799](https://github.com/nandeesh2799)
