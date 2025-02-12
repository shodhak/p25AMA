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

# Add process description with multiple lines
st.caption("""💡 Process: The model generates three responses to each query and synthesizes final response from those responses.
\n📚 Content: The model may use outside info to enhance the answer, but when it does it will mention that.
\n💡 TIP: Writing full form instead of acronyms gives better results. Say National Institutes of Health if NIH doesn't work.""")

# Check if Enter was pressed or button clicked
if get_answer or st.session_state.submit:
    st.session_state.submit = False  # Reset the submit flag
    if query:
        with st.spinner("🤖 Generating answer..."):
            try:
                response = requests.get("http://127.0.0.1:8000/query/", params={"query": query})
                if response.status_code == 200:
                    answer = response.json().get("answer", "No response received.")
                    st.success("✅ Answer:")
                    st.write(answer)
                else:
                    st.error("❌ Error querying the API. Please check FastAPI logs.")
            except requests.exceptions.ConnectionError:
                st.error("⚠️ FastAPI server is not running. Please start it first.")
    else:
        st.warning("⚠️ Please enter a question before clicking 'Get Answer'.")

st.markdown("---")
st.write("Developed using **Facebook LLaMA 3.2** 🚀")