# Iris Flower Species Classification (End-to-End)

This project trains and compares multiple ML models on the classic Iris dataset, generates EDA visualizations, saves the best model (`pickle`), builds a modern Streamlit web app for inference, and auto-creates a PowerPoint presentation.

## Project structure

```
.
├── app/                    # Streamlit app
├── assets/images/          # Real flower images (local)
├── data/                   # Dataset (CSV exported from sklearn)
├── models/                 # Saved model + metadata
├── notebooks/              # EDA + training pipeline (python script)
└── outputs/
    ├── plots/              # Saved plots for PPT
    └── ppt/                # Auto-generated PPT + assets
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run training + EDA (creates model, plots, metrics, PPT)

```bash
python notebooks/train_eda_and_ppt.py
```

Artifacts produced:
- `data/iris.csv`
- `outputs/plots/*.png`
- `models/best_model.pkl`
- `models/metadata.json`
- `outputs/ppt/Iris_Flower_Species_Classification.pptx`

## Run the Streamlit app

```bash
streamlit run app/app.py
```

## Notes
- The Streamlit UI hides the default menu, deploy button, header, and footer via custom CSS.
- Species images are stored locally in `assets/images/` (downloaded from Wikimedia Commons).

