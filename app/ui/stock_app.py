import streamlit as st
import requests
import os
import sys

# Setup project root paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

core_path = os.path.join(project_root, "core")
if core_path not in sys.path:
    sys.path.insert(0, core_path)

from dotenv import load_dotenv
from core.settings import settings

load_dotenv()

BASE_URL = settings.BASE_URL

# Page config
st.set_page_config(page_title="📈 Stock AI Prediction", layout="wide")

# CSS to fix sidebar height and style
st.markdown("""
<style>
.sidebar .block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    height: 90vh;  /* full-ish height */
    overflow-y: auto;
}
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    height: 90vh;
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)

# Use Streamlit's sidebar for left panel input (fixed width and full height)
with st.sidebar:
    st.header("🔎 Query")
    ticker = st.text_input("Enter Stock Ticker (e.g., RELIANCE, TCS):").upper()
    predict = st.button("🔍 Predict Stock")

    # Show some quick analyzing indicators here if data exists
    # (we will do it in main area after fetching data)

# Main area for result display
st.title("🧠 Stock AI with GenAI Summary")

if predict and ticker:
    with st.spinner("Analyzing..."):
        try:
            # Request stock data and prediction summary
            response = requests.get(f"{BASE_URL}/stock/{ticker}")
            response.raise_for_status()
            data = response.json()

            # Show summary in a container with fixed height and scrollable if needed
            with st.container():
                st.subheader(f"📊 Prediction for {ticker}")

                # Quick price difference with coloring
                current_price = data['current_price']
                predicted_price = data['predicted_price']
                price_diff = predicted_price - current_price
                diff_pct = (price_diff / current_price) * 100 if current_price else 0

                color = "green" if price_diff > 0 else ("red" if price_diff < 0 else "orange")
                sign = "+" if price_diff > 0 else ""

                st.markdown(f"""
                - **Current Price:** ₹ {current_price}
                - **Predicted Price:** ₹ {predicted_price:.2f}
                - **Price Change:** <span style='color:{color}; font-weight:bold;'>{sign}{price_diff:.2f} ({sign}{diff_pct:.2f}%)</span>
                - **Recommendation:** {data['recommendation']}
                """, unsafe_allow_html=True)

                st.markdown("### 📄 GenAI Summary")
                st.info(data['summary'])

            # Show chart next to summary with fixed height
            chart_url = f"{BASE_URL}/stock/{ticker}/chart"
            chart_response = requests.get(chart_url)

            if chart_response.status_code == 200:
                st.image(chart_response.content, caption="📈 1-Year Stock Chart with ML Prediction", use_container_width=True)
            else:
                st.warning("⚠️ Chart not available.")

        except Exception as e:
            st.error(f"❌ Failed to get prediction: {str(e)}")

else:
    st.info("Enter a stock ticker and click 'Predict Stock' to get started.")
