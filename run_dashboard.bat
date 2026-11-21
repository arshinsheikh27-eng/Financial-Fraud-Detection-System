@echo off
python -m pip install -r requirements-app.txt
python -m streamlit run dashboard/app.py
