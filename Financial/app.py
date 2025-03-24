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
st.set_page_config(page_title="Revenue Forecasting Agent", page_icon="📈", layout="wide")
st.title("📈 AI Revenue Forecasting Agent with Prophet")

# File upload
uploaded_file = st.file_uploader("Upload your Excel file (with 'Date' and 'Revenue' columns)", type=["xlsx"])

if uploaded_file:
    # Read the Excel file
    df = pd.read_excel(uploaded_file)
    
    st.subheader("📄 Uploaded Data Preview")
    st.write(df.head())

    # Data validation
    if "Date" not in df.columns or "Revenue" not in df.columns:
        st.error("Uploaded file must contain 'Date' and 'Revenue' columns.")
        st.stop()

    # Preprocessing
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Revenue']].rename(columns={'Date': 'ds', 'Revenue': 'y'})

    # Prophet Forecasting
    model = Prophet()
    model.fit(df)

    # User selects forecast horizon
    periods = st.slider("Select forecast horizon (months)", 1, 24, 6)
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)

    # Plot Forecast
    st.subheader("🔮 Forecast Plot")
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
    You are a Financial Analyst. Analyze the following forecasted revenue data and provide:
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

