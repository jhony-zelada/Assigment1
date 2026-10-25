import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.title("Student Performance Predictor")
st.caption("Upload a CSV file, pick an input and a target, and see what a straight line predicts.")

# The shape is the same as the first app.
# The user uploads a file, and everything else sits inside an if.
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # This time we need two numeric columns:
    # one to use as the input and one to predict.
    # Two select boxes let the user choose.
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        st.error("This file needs at least two numeric columns.")
        st.stop()

    col1, col2 = st.columns(2)
    x_col = col1.selectbox("Input: what you know", numeric_cols)
    y_options = [c for c in numeric_cols if c != x_col]
    y_col = col2.selectbox("Target: what you want to predict", y_options)

    data = df[[x_col, y_col]].dropna()
    if len(data) < 5:
        st.error("Need at least 5 rows with values in both columns.")
        st.stop()

    # Then the model.
    # We keep the two columns, fit the line,
    # and draw the dots with the line through them.
    x = data[[x_col]]
    y = data[y_col]
    model = LinearRegression()
    model.fit(x, y)

    fig = px.scatter(data, x=x_col, y=y_col, title=f"{y_col} versus {x_col}")
    ends = pd.DataFrame({x_col: [data[x_col].min(), data[x_col].max()]})
    fig.add_scatter(x=ends[x_col], y=model.predict(ends), mode="lines", name="Model")
    st.plotly_chart(fig, width="stretch")

    # Two numbers go underneath: the slope, and R squared.
    m1, m2 = st.columns(2)
    m1.metric(f"Each extra {x_col} adds", f"{model.coef_[0]:,.2f}")
    m2.metric("R2 on this data", f"{model.score(x, y):.2f}")

    # The last part is the prediction.
    # The user types a value, and the app reads the answer from the model.
    # Notice the table we give to predict: it has the same column name
    # as the training data. A model expects inputs in the same form
    # it was trained on.
    st.subheader("Try it")
    value = st.number_input(f"Enter a value for {x_col}", value=float(data[x_col].mean()))
    prediction = model.predict(pd.DataFrame({x_col: [value]}))[0]
    st.metric(f"Predicted {y_col}", f"{prediction:,.0f}")

    # The last lines are a warning.
    # If the user types a value outside the range of the data, the app says so.
    # Our visitors run from 2,950 to 5,200. A line does not know where its data
    # stopped. Ask it about 50,000 visitors, and it will happily give you
    # a confident number. A good app tells the user when to trust it less.
    low, high = data[x_col].min(), data[x_col].max()
    if value < low or value > high:
        st.warning(f"This value is outside the range the model learned from ({low:,.0f} to {high:,.0f}). Treat the number with caution.")
else:
    st.info("Upload a CSV file to get started. It needs at least two numeric columns.")
