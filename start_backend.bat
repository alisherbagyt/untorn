@echo off
echo ============================================================
echo  UNTORN — Starting FastAPI Backend (port 8000)
echo ============================================================
cd /d C:\dev\untorn
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
