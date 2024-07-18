import streamlit as st
from cover_page import page_cover
from disclaimer_page import page_disclaimer
from zscore_analysis_page import page_zscore_analysis

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Cover Page", "Disclaimer-Disclosure", "Z-Score Analysis"])

    PAGES = {
        "Cover Page": page_cover,
        "Disclaimer-Disclosure": page_disclaimer,
        "Z-Score Analysis": page_zscore_analysis,
    }

    # Execute the selected page function
    page = PAGES[selection]
    page()

if __name__ == "__main__":
    main()