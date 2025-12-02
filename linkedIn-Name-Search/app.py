import streamlit as st
from backend import LinkedInSearchAgent
import os
from dotenv import load_dotenv

load_dotenv()


# Initialize agent
agent = LinkedInSearchAgent(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)


st.set_page_config(page_title="LinkedIn Search Agent", layout="centered")


# -----------------------------------------
# UI HEADER
# -----------------------------------------
st.title("🔎 LinkedIn Intelligence Search")
st.caption("Search people, companies, colleges — powered by Tavily + Gemini")


# -----------------------------------------
# Search Type Selection
# -----------------------------------------
search_type = st.radio(
    "Search Type",
    ["person", "company"],
    horizontal=True
)


# -----------------------------------------
# Input Fields
# -----------------------------------------
name = st.text_input(
    "Name",
    placeholder="Enter name... (e.g., John Smith, Google, Amazon)"
)

city = st.text_input("City (optional)", placeholder="Surat, Mumbai, Bengaluru...")
role = st.text_input("Role (optional)", placeholder="Software Engineer, AI Lead...")
college = st.text_input("College (optional)", placeholder="NIT, IIT, SSASIT...")
company = st.text_input("Company (optional)", placeholder="Infosys, AppStoneLab...")


# -----------------------------------------
# Search Button
# -----------------------------------------
if st.button("Search 🔍", use_container_width=True):
    if not name.strip():
        st.error("Name field cannot be empty.")
        st.stop()

    with st.spinner("Fetching LinkedIn data..."):
        result = agent.search(
            type=search_type,
            name=name,
            city=city if city else None,
            role=role if role else None,
            college=college if college else None,
            company=company if company else None
        )

    # -----------------------------------------
    # Response Handling
    # -----------------------------------------
    if result["status"] == "Error":
        st.error(result["output"])
    else:
        output = result["output"]["output"] if isinstance(result["output"], dict) else result["output"]

        if not output:
            st.warning("No results returned.")
        else:
            st.subheader("📌 Search Results")
            st.write(output.replace("\n", "  \n"))
