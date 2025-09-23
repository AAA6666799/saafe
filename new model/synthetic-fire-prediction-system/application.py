import streamlit as st
import os

# For EB health check
def main():
    st.title("Fire Detection Dashboard")
    st.write("Dashboard is running!")

if __name__ == "__main__":
    # Check if this is a health check request
    if os.environ.get('REQUEST_URI') == '/health':
        print("Healthy")
    else:
        main()
