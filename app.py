import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.title("Student Performance Predictor")
st.caption(
    "Upload Student_Performance.csv, explore a column, then fit a line that predicts "
    "Performance Index from two numeric inputs."
)

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ---------- Preview ----------
    st.subheader("Preview")
    st.dataframe(df.head(10), use_container_width=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 3:
        st.error("This app needs at least three numeric columns: two inputs and one target.")
        st.stop()

    # ---------- Explore a column ----------
    st.subheader("Explore a column")
    default_chart_col = numeric_cols[-1]  # Performance Index in the sample file
    chart_col = st.selectbox("Column to chart", numeric_cols, index=numeric_cols.index(default_chart_col))
    chart_type = st.selectbox("Chart type", ["Histogram", "Line", "Scatter"])

    if chart_type == "Histogram":
        fig = px.histogram(df, x=chart_col, title=f"Distribution of {chart_col}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Line":
        # Values in row order (good for time-ordered or sequential data).
        fig = px.line(df, y=chart_col, title=f"{chart_col} in row order")
        st.plotly_chart(fig, use_container_width=True)

    else:  # Scatter needs a second numeric column for the x-axis
        x_options = [c for c in numeric_cols if c != chart_col]
        if not x_options:
            st.warning("Need at least two numeric columns to draw a scatter plot.")
        else:
            x_col_chart = st.selectbox("X axis for scatter", x_options)
            fig = px.scatter(df, x=x_col_chart, y=chart_col,
                             title=f"{chart_col} versus {x_col_chart}")
            st.plotly_chart(fig, use_container_width=True)

    # ---------- Fit a linear regression ----------
    st.subheader("Fit a linear regression")

    default_target = "Performance Index" if "Performance Index" in numeric_cols else numeric_cols[-1]
    target = st.selectbox("Target (what you want to predict)", numeric_cols,
                          index=numeric_cols.index(default_target))

    input_options = [c for c in numeric_cols if c != target]
    input_a = st.selectbox("Input 1", input_options)
    input_b = st.selectbox("Input 2", [c for c in input_options if c != input_a])

    data = df[[input_a, input_b, target]].dropna()
    if len(data) < 5:
        st.error("Need at least 5 rows with values in all three columns.")
        st.stop()

    # IMPORTANT: X keeps the same column names AND order the model will see at predict time.
    X = data[[input_a, input_b]]
    y = data[target]

    model = LinearRegression()
    model.fit(X, y)

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Effect of {input_a}", f"{model.coef_[0]:,.3f}")
    m2.metric(f"Effect of {input_b}", f"{model.coef_[1]:,.3f}")
    m3.metric("R² on this data", f"{model.score(X, y):.3f}")

    # ---------- Predict ----------
    st.subheader("Try it")
    c1, c2 = st.columns(2)
    val_a = c1.number_input(f"Enter a value for {input_a}", value=float(data[input_a].mean()))
    val_b = c2.number_input(f"Enter a value for {input_b}", value=float(data[input_b].mean()))

    # Same column names and order as training: {input_a, input_b}.
    new_row = pd.DataFrame({input_a: [val_a], input_b: [val_b]})
    prediction = model.predict(new_row)[0]
    st.metric(f"Predicted {target}", f"{prediction:,.2f}")

    # Warn when the user leaves the range the model was trained on.
    lo_a, hi_a = data[input_a].min(), data[input_a].max()
    lo_b, hi_b = data[input_b].min(), data[input_b].max()
    if not (lo_a <= val_a <= hi_a) or not (lo_b <= val_b <= hi_b):
        st.warning(
            f"One or both values are outside the range the model learned from "
            f"({input_a}: {lo_a:,.0f}–{hi_a:,.0f}, {input_b}: {lo_b:,.0f}–{hi_b:,.0f}). "
            f"Treat this prediction with caution."
        )

    # ---------- Required one sentence ----------
    st.info(
        "This model estimates a student's Performance Index from two numeric factors; "
        "when it is wrong, decisions taken from it (e.g. spotting at-risk students) can be off, "
        "so always sanity-check the prediction against the observed range."
    )

else:
    st.info("Upload a CSV file to get started. It needs at least three numeric columns.")
