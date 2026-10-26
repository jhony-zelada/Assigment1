import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

# ---------- Session state ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- Header ----------
st.title("🎓 Student Performance Predictor")
st.caption(
    "Upload a CSV with **at least three numeric columns**. Explore a column, fit a "
    "straight line that predicts one target from two inputs, and try predictions."
)

# ---------- Sidebar: data source ----------
with st.sidebar:
    st.header("📁 Data source")
    uploaded = st.file_uploader("Upload a CSV", type="csv")
    use_sample = st.checkbox(
        "Use built-in `Student_Performance.csv`",
        value=False,
        help="Works if the CSV is placed next to `app.py` in the repo.",
    )

    df = None
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
    elif use_sample:
        try:
            df = pd.read_csv("Student_Performance.csv")
        except FileNotFoundError:
            st.error("`Student_Performance.csv` not found next to `app.py`.")

    if df is not None:
        st.success(f"Loaded: **{len(df):,}** rows × **{df.shape[1]}** columns")

    st.divider()
    if st.button("🧹 Reset prediction history", use_container_width=True):
        st.session_state.history.clear()
        st.toast("Prediction history cleared.")

if df is None:
    st.info("👈 Upload a CSV in the sidebar (or tick the sample option) to start.")
    st.stop()

# ---------- Column validation ----------
numeric_cols = df.select_dtypes(include="number").columns.tolist()
usable = [c for c in numeric_cols if df[c].notna().sum() >= 5 and df[c].nunique() > 1]

if len(usable) < 3:
    st.error(
        f"This app needs at least **three usable numeric columns** "
        f"(two inputs + one target). Usable: {usable or 'none'}. "
        f"Numeric in file: {numeric_cols or 'none'}."
    )
    st.stop()

# =====================================================================
# TABS
# =====================================================================
tab_preview, tab_explore, tab_predict = st.tabs(["📄 Preview", "📊 Explore", "🔮 Predict"])

# ---------- Preview ----------
with tab_preview:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows", f"{len(df):,}")
    k2.metric("Columns", df.shape[1])
    k3.metric("Numeric columns", len(numeric_cols))
    k4.metric("Missing values", int(df.isna().sum().sum()))

    st.subheader("First 20 rows")
    st.dataframe(df.head(20), use_container_width=True, height=340)

    with st.expander("📋 Column summary (numeric)"):
        st.dataframe(
            df[usable].describe().T[["count", "mean", "std", "min", "max"]].round(2),
            use_container_width=True,
        )

# ---------- Explore ----------
with tab_explore:
    st.subheader("Chart a column")

    e1, e2 = st.columns([2, 1])
    chart_col = e1.selectbox(
        "Column to chart", usable,
        index=len(usable) - 1,
        help="Only numeric columns are shown.",
    )
    chart_type = e2.selectbox("Chart type", ["Histogram", "Line", "Scatter"])

    fig = None
    if chart_type == "Histogram":
        fig = px.histogram(df, x=chart_col, nbins=40,
                           title=f"Distribution of {chart_col}",
                           color_discrete_sequence=["#4C78A8"])
        fig.update_layout(bargap=0.05)
    elif chart_type == "Line":
        fig = px.line(df, y=chart_col, title=f"{chart_col} in row order")
        fig.update_traces(line_color="#F58518")
    else:
        others = [c for c in usable if c != chart_col]
        if not others:
            st.warning("Need at least two usable numeric columns for a scatter plot.")
        else:
            x_chart = st.selectbox("X axis for scatter", others)
            fig = px.scatter(df, x=x_chart, y=chart_col,
                             title=f"{chart_col} versus {x_chart}",
                             opacity=0.7)
    if fig is not None:
        fig.update_layout(margin=dict(t=50, b=10), height=440)
        st.plotly_chart(fig, use_container_width=True)

# ---------- Predict ----------
with tab_predict:
    st.subheader("Fit a line and predict")

    c1, c2, c3 = st.columns(3)
    default_target = "Performance Index" if "Performance Index" in usable else usable[-1]
    target = c1.selectbox(
        "🎯 Target (what to predict)", usable,
        index=usable.index(default_target),
        help="The column the model will estimate.",
    )
    inputs_pool = [c for c in usable if c != target]
    input_a = c2.selectbox("🔑 Input 1", inputs_pool, help="First feature.")
    input_b = c3.selectbox(
        "🔑 Input 2", [c for c in inputs_pool if c != input_a],
        help="Second feature (must differ from Input 1).",
    )

    data = df[[input_a, input_b, target]].dropna()
    if len(data) < 5:
        st.error("Need at least 5 rows with non-missing values in all three columns.")
        st.stop()

    # IMPORTANT: the model sees the same column names AND order at fit and predict time.
    X = data[[input_a, input_b]]
    y = data[target]

    with st.spinner("Fitting the model…"):
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)

    r2   = r2_score(y, y_pred)
    mae  = mean_absolute_error(y, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    resid_std = float(np.std(y - y_pred, ddof=1))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R²", f"{r2:.3f}", help="1 = perfect fit, 0 = no better than the mean.")
    m2.metric("MAE", f"{mae:,.2f}", help="Mean absolute error on this data.")
    m3.metric("RMSE", f"{rmse:,.2f}", help="Root-mean-square error.")
    m4.metric("Residual σ", f"{resid_std:,.2f}", help="Std-dev of the residuals.")

    with st.expander("📐 Model coefficients"):
        coefs = pd.DataFrame({
            "Feature": ["(intercept)"] + list(X.columns),
            "Coefficient": [model.intercept_] + list(model.coef_),
        }).round(4)
        st.dataframe(coefs, use_container_width=True, hide_index=True)

    st.subheader("Try it")
    unlock = st.toggle(
        "Allow values outside the training range",
        value=False,
        help="Unlocked values trigger a warning — a straight line extrapolates blindly.",
    )

    def input_widget(label, col):
        lo = float(data[col].min()); hi = float(data[col].max()); mean = float(data[col].mean())
        if unlock or np.isclose(lo, hi):
            return st.number_input(label, value=mean, format="%.2f",
                                   help=f"Training range: {lo:,.2f} – {hi:,.2f}")
        is_int = lo.is_integer() and hi.is_integer()
        step = 1.0 if is_int else max((hi - lo) / 200.0, 0.01)
        return st.slider(label, lo, hi, mean, step=step,
                         help=f"Training range: {lo:,.2f} – {hi:,.2f}")

    p1, p2 = st.columns(2)
    with p1:
        val_a = input_widget(input_a, input_a)
    with p2:
        val_b = input_widget(input_b, input_b)

    # Same column names and order as X.
    new_row = pd.DataFrame({input_a: [val_a], input_b: [val_b]})
    pred = float(model.predict(new_row)[0])

    pc1, pc2 = st.columns([1, 2])
    pc1.metric(f"🎯 Predicted {target}", f"{pred:,.2f}")
    if pc2.button("➕ Save this prediction", use_container_width=True):
        st.session_state.history.append({
            input_a: val_a, input_b: val_b, f"predicted {target}": round(pred, 2),
        })
        st.toast("Saved.")

    lo_a = float(data[input_a].min()); hi_a = float(data[input_a].max())
    lo_b = float(data[input_b].min()); hi_b = float(data[input_b].max())
    if not (lo_a <= val_a <= hi_a) or not (lo_b <= val_b <= hi_b):
        st.warning(
            f"⚠️ One or both values are outside the training range "
            f"({input_a}: {lo_a:,.2f}–{hi_a:,.2f}, {input_b}: {lo_b:,.2f}–{hi_b:,.2f}). "
            "A line extrapolates blindly — treat this number with caution."
        )

    with st.expander("🖼️ See the prediction on a chart"):
        sweep = pd.DataFrame({
            input_a: np.linspace(lo_a, hi_a, 100),
            input_b: [val_b] * 100,
        })
        line = model.predict(sweep)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data[input_a], y=data[target], mode="markers",
                                 name="Data", marker=dict(opacity=0.4)))
        fig.add_trace(go.Scatter(x=sweep[input_a], y=line, mode="lines",
                                 name=f"Fit at {input_b}={val_b:,.2f}",
                                 line=dict(width=3)))
        fig.add_trace(go.Scatter(x=[val_a], y=[pred], mode="markers",
                                 name="Your prediction",
                                 marker=dict(size=16, color="crimson", symbol="star")))
        fig.update_layout(
            title=f"{target} vs {input_a} (with {input_b} held at {val_b:,.2f})",
            height=420, margin=dict(t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    if st.session_state.history:
        st.subheader("🗒️ Prediction history")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download history (CSV)",
                           hist_df.to_csv(index=False).encode(),
                           file_name="predictions.csv", mime="text/csv")

# ---------- Required one-sentence note (always visible, below the tabs) ----------
st.divider()
st.info(
"This model is used to guide actions on variables such as 'Hours Studied' and 'Sleep Hours' to"
 "improve student performance. If the prediction is wrong, the measures taken could have no"
 "effect on performance, which would lead to a waste money and time."
)
