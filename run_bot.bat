@echo off
:: TitanBrain — Bot permanente de Telegram (procesa respuestas de botones)
:: Corre al iniciar Windows y se mantiene activo 24/7
cd /d "C:\Users\Luis Robinson\Desktop\DeepQuant"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

if not exist logs mkdir logs

:loop
echo [%date% %time%] Iniciando bot... >> logs\bot_log.txt
".venv\Scripts\python.exe" alerts_live.py bot >> logs\bot_log.txt 2>&1
echo [%date% %time%] Bot cerrado inesperadamente. Reiniciando en 10s... >> logs\bot_log.txt
timeout /t 10 /nobreak >nul
goto loop
