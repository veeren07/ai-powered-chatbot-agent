"""
run_app.py - Python launcher for the AI-Powered PDF Chatbot.

Runs Streamlit directly within Python, bypassing Windows Application Control /
AppLocker policy blocks (os error 4551) on standalone wrapper .exe files.

Usage:
    uv run python run_app.py
    # or
    uv run python -m streamlit run app.py
"""

import sys
from streamlit.web import cli as stcli

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "app.py"]
    sys.exit(stcli.main())

