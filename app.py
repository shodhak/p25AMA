import streamlit as st
import requests

# Set page title
st.set_page_config(page_title="PDF Query App", layout="wide")

st.title("📄 Project 2025 AMA")

# Initialize session state for submit flag if it doesn't exist
if "submit" not in st.session_state:
    st.session_state.submit = False

# Function to handle query submission
def submit_query():
    st.session_state.submit = True

# Input field for user query with on_change handler
query = st.text_input("🔍 Enter your question:", key="query_input", on_change=submit_query)

# Add Get Answer button
get_answer = st.button("Get Answer")

# API URL - Change this if deployed on Railway
#API_URL = "http://127.0.0.1:8000/query/"  # For local testing
API_URL = "https://project2025.up.railway.app/query/"  # Uncomment for Railway deployment

# Add process description with multiple lines
st.caption("""💡 Process: The model generates three responses to each query from OpenAI, Facebook Llama and Deepseek R1, and synthesizes a final response from those answers.
\n📚 Content: The model may use outside info to enhance the answer, but when it does it will mention that.
\n💡 TIP: Writing full form instead of acronyms gives better results. Say National Institutes of Health if NIH doesn't work.""")

# Check if Enter was pressed or button clicked
if get_answer or st.session_state.submit:
    st.session_state.submit = False  # Reset the submit flag
    if query:
        with st.spinner("🔎 Searching for the best answer... Please wait."):
            try:
                response = requests.get(API_URL, params={"query": query}, timeout=15)
                if response.status_code == 200:
                    answer = response.json().get("answer", "No response received.")
                    st.success("✅ Answer:")
                    st.write(answer)
                else:
                    st.error(f"❌ API Error: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("⚠️ FastAPI server is not running. Please start it first.")
            except requests.exceptions.Timeout:
                st.error("⏳ The request took too long. Please try again later.")
            except requests.exceptions.RequestException as e:
                st.error(f"⚠️ Request failed: {e}")
    else:
        st.warning("⚠️ Please enter a question before clicking 'Get Answer'.")

st.markdown("---")
st.write("Developed using **OpenAI GPT-4o, Facebook LLaMA 3.2, and DeepSeek R1** 🚀")