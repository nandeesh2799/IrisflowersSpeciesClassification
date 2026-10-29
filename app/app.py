from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parents[0]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils import MODEL_PATH, SPECIES, load_model, make_input_df, predict


def inject_global_css() -> None:
    st.markdown(
        """
<style>
/* Required: hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Modern iris-themed styling */
.stApp {
  background:
    radial-gradient(900px 500px at 15% 10%, rgba(124, 77, 255, 0.22), rgba(0,0,0,0)),
    radial-gradient(850px 520px at 85% 20%, rgba(0, 210, 190, 0.18), rgba(0,0,0,0)),
    radial-gradient(650px 520px at 55% 95%, rgba(255, 88, 168, 0.12), rgba(0,0,0,0)),
    linear-gradient(180deg, #0b1220 0%, #0a1020 55%, #070c18 100%);
}

/* Force readable typography on dark background */
html, body, [data-testid="stAppViewContainer"] {
  color: rgba(242, 246, 255, 0.94) !important;
}
h1, h2, h3, h4, h5, h6 {
  color: rgba(250, 252, 255, 0.98) !important;
}
label, p, li, span, div {
  color: rgba(242, 246, 255, 0.92);
}
[data-testid="stCaptionContainer"] {
  color: rgba(242, 246, 255, 0.72) !important;
}

.app-wrap { max-width: 980px; margin: 0 auto; padding: 0.75rem 0.25rem 1.25rem; }

.hero {
  position: relative;
  padding: 1.25rem 1.25rem;
  border-radius: 20px;
  overflow: hidden;
  background: linear-gradient(135deg, rgba(124, 77, 255, 0.22), rgba(0, 210, 190, 0.16));
  border: 1px solid rgba(255,255,255,0.10);
  box-shadow: 0 18px 60px rgba(0,0,0,0.35);
}
.hero:before {
  content: "";
  position: absolute;
  inset: 0;
  background-image:
    radial-gradient(circle at 25% 25%, rgba(255,255,255,0.10) 0 1px, transparent 1px 100%),
    radial-gradient(circle at 75% 40%, rgba(255,255,255,0.08) 0 1px, transparent 1px 100%);
  opacity: 0.55;
}
.hero-inner { position: relative; }
.hero-title { font-size: 2.25rem; font-weight: 860; letter-spacing: -0.03em; margin: 0; line-height: 1.08; }
.hero-sub { margin-top: .35rem; opacity: .88; font-size: 1.02rem; }
.chip-row { margin-top: .8rem; display: flex; gap: .5rem; flex-wrap: wrap; }
.chip {
  display: inline-flex;
  align-items: center;
  gap: .4rem;
  padding: .32rem .62rem;
  border-radius: 999px;
  background: rgba(255,255,255,0.10);
  border: 1px solid rgba(255,255,255,0.14);
  font-size: .88rem;
}

.glass {
  padding: 1.05rem 1.05rem;
  border-radius: 18px;
  background: rgba(255,255,255,0.10);
  border: 1px solid rgba(255,255,255,0.16);
  box-shadow: 0 10px 40px rgba(0,0,0,0.28);
  backdrop-filter: blur(10px);
}
.glass:hover { border-color: rgba(255,255,255,0.18); }

.badge {
  display: inline-flex;
  padding: .22rem .55rem;
  border-radius: 999px;
  background: rgba(124,77,255,0.20);
  border: 1px solid rgba(124,77,255,0.45);
  font-size: .85rem;
}

.metric-pill {
  display: inline-block;
  padding: .30rem .55rem;
  border-radius: 999px;
  background: rgba(0,210,190,0.14);
  border: 1px solid rgba(0,210,190,0.40);
  margin-right: .45rem;
  margin-bottom: .35rem;
  font-size: .90rem;
}

/* Make sliders feel more premium */
div[data-baseweb="slider"] > div { padding-top: 0.15rem; padding-bottom: 0.15rem; }
div[data-baseweb="slider"] [role="slider"] { box-shadow: 0 0 0 6px rgba(124, 77, 255, 0.12); }
div[data-baseweb="slider"] * { color: rgba(250, 252, 255, 0.95) !important; }

/* Tighten tab header spacing */
button[data-baseweb="tab"] { font-size: 0.98rem; }
button[data-baseweb="tab"] * { color: rgba(250, 252, 255, 0.92) !important; }

/* Make inputs + alerts readable */
[data-testid="stAlert"] * { color: rgba(250,252,255,0.98) !important; }
</style>
""",
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


def main() -> None:
    st.set_page_config(
        page_title="Iris Flower Species Classification",
        page_icon="Iris",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    inject_global_css()

    st.markdown('<div class="app-wrap">', unsafe_allow_html=True)
    st.markdown(
        """
<div class="hero">
  <div class="hero-inner">
    <div class="hero-title">Iris Flower Species Classification</div>
    <div class="hero-sub">A fast, interactive classifier powered by a tuned scikit-learn model.</div>
    <div class="chip-row">
      <div class="chip">Dataset: <b>Iris</b> (150 rows)</div>
      <div class="chip">Classes: <b>3</b> species</div>
      <div class="chip">Inputs: <b>4</b> measurements</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    with st.spinner("Initializing classifier pipeline..."):
        model = get_model()

    tab_predict, tab_species = st.tabs(["Predict", "Species gallery"])

    with tab_predict:
        c1, c2 = st.columns([1.05, 0.95], gap="large")
        with c1:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.subheader("Input measurements")
            st.caption("Tip: petal measurements usually drive the strongest separation.")

            with st.form("predict_form", border=False):
                sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
                sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
                petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
                petal_width = st.slider("Petal Width (cm)", 0.1, 2.6, 1.3, 0.1)
                do_predict = st.form_submit_button("Predict species", type="primary", use_container_width=True)

            X = make_input_df(sepal_length, sepal_width, petal_length, petal_width)
            st.markdown(
                f"""
<div class="chip-row">
  <div class="chip">Sepal: <b>{sepal_length:.1f}</b> × <b>{sepal_width:.1f}</b> cm</div>
  <div class="chip">Petal: <b>{petal_length:.1f}</b> × <b>{petal_width:.1f}</b> cm</div>
</div>
""",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.subheader("Prediction")

            if do_predict:
                label, proba = predict(model, X)
                info = SPECIES.get(label.lower())

                if info is None:
                    st.warning(f"Predicted label: {label}")
                else:
                    st.markdown(f"### {info.common_name}  <span class='badge'>predicted</span>", unsafe_allow_html=True)
                    if info.image_path.exists():
                        st.image(str(info.image_path), use_container_width=True)
                    st.write(info.description)

                    if proba:
                        st.write("")
                        st.markdown("**Confidence**")
                        for k, v in sorted(proba.items(), key=lambda kv: kv[1], reverse=True):
                            st.markdown(f"<span class='metric-pill'>{k}: {v:.2%}</span>", unsafe_allow_html=True)
                            st.progress(float(v))
            else:
                st.info("Adjust inputs, then click **Predict species**.")

            st.markdown("</div>", unsafe_allow_html=True)

    with tab_species:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("Species gallery")
        st.caption("Real photos stored locally in `assets/images/`.")
        g1, g2, g3 = st.columns(3, gap="medium")
        cols = [g1, g2, g3]
        for col, key in zip(cols, ["setosa", "versicolor", "virginica"]):
            info = SPECIES[key]
            with col:
                st.markdown(f"**{info.common_name}**")
                if info.image_path.exists():
                    st.image(str(info.image_path), use_container_width=True)
                st.write(info.description)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.caption("Images in `assets/images/` are from Wikimedia Commons and used under their respective licenses.")
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

