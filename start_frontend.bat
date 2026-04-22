@echo off
echo ============================================================
echo  UNTORN — Starting Next.js Frontend (port 3000)
echo ============================================================
cd /d C:\dev\untorn\frontend
call npm run dev -- --hostname 0.0.0.0
