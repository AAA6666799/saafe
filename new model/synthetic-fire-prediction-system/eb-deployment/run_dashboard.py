import subprocess
import sys
import os
import time

# Install required packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements-dashboard.txt"])

# Run Streamlit dashboard
if __name__ == '__main__':
    os.system("streamlit run application.py --server.port 8501 --server.address 0.0.0.0")