@echo off
python -m pip install -r requirements-app.txt
python -m uvicorn backend.main:app --reload
