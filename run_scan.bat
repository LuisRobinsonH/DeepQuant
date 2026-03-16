@echo off
:: TitanBrain — Scan horario de señales ASX
:: Llamado automáticamente por Task Scheduler cada hora
cd /d "C:\Users\Luis Robinson\Desktop\DeepQuant"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

if not exist logs mkdir logs

:: Log con timestamp
echo. >> logs\scan_log.txt
echo ===== %date% %time% ===== >> logs\scan_log.txt

".venv\Scripts\python.exe" alerts_live.py scan >> logs\scan_log.txt 2>&1
