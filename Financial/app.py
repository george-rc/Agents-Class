import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import os
from dotenv import load_dotenv
from groq import Groq

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# 🎨 Streamlit UI Styling
st.set_page_config(page_title="Forecasting Agent - Revenue | Expense | Profit", page_icon="📈", layout="wide")
st.title("📈 AI Forecasting Agent (Revenue / Expense / Profit) with Prophet")

# File upload
uploaded_file = st.file_uploader("Upload your Excel file (with 'Date' and any of 'Revenue', 'Expense', 'Profit' columns)", type=["xlsx"])

if uploaded_file:
    # Read the Excel file
    df = pd.read_excel(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.write(df.head())

    # Check required 'Date' column
    if "Date" not in df.columns:
        st.error("Uploaded file must contain a 'Date' column.")
        st.stop()

    # Detect available financial columns
    available_targets = [col for col in ['Revenue', 'Expense', 'Profit'] if col in df.columns]
    if not available_targets:
        st.error("Your dataset must contain at least one of these columns: 'Revenue', 'Expense', or 'Profit'.")
        st.stop()

    # Let user select which target to forecast
    target = st.selectbox("Select the metric to forecast", available_targets)

    # Preprocessing
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', target]].rename(columns={'Date': 'ds', target: 'y'})

    # Prophet Forecasting
    model = Prophet()
    model.fit(df)

    # Forecast horizon selection
    periods = st.slider("Select forecast horizon (months)", 1, 24, 6)
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)

    # Plot Forecast
    st.subheader(f"🔮 Forecast Plot for {target}")
    fig = plot_plotly(model, forecast)
    st.plotly_chart(fig)

    # Show Forecast Data
    st.subheader("📊 Forecast Data")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods))

    # AI Commentary with GROQ API
    st.subheader("🤖 AI-Generated Forecast Commentary")
    data_for_ai = forecast[['ds', 'yhat']].tail(periods).to_json()

    client = Groq(api_key=GROQ_API_KEY)
    prompt = f"""
    You are a Financial Analyst. Analyze the following forecasted {target.lower()} data and provide:
    - Key trends observed in the forecast.
    - Risks or uncertainties.
    - Actionable insights for business planning.

    Dataset (JSON):
    {data_for_ai}
    """

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a financial forecasting expert."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-8b-8192",
    )
    ai_commentary = response.choices[0].message.content
    st.write(ai_commentary)

else:
    st.info("👆 Please upload an Excel file to begin.")


