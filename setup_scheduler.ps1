# setup_scheduler.ps1 — Configura Task Scheduler para TitanBrain
# Ejecutar UNA vez como Administrador:
#   Right-click → "Run as administrator" o:
#   powershell -ExecutionPolicy Bypass -File setup_scheduler.ps1

$WorkDir = "C:\Users\Luis Robinson\Desktop\DeepQuant"
$BotBat  = Join-Path $WorkDir "run_bot.bat"
$ScanBat = Join-Path $WorkDir "run_scan.bat"

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  TitanBrain — Configurando Task Scheduler" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Crear carpeta de logs si no existe
$LogDir = Join-Path $WorkDir "logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
    Write-Host "[OK] Carpeta logs/ creada" -ForegroundColor Green
}


# ── TAREA 1: Bot permanente (arranca con Windows) ─────────────────────
$TaskName1 = "TitanBrain_Bot"
# Eliminar si ya existe
if (Get-ScheduledTask -TaskName $TaskName1 -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName1 -Confirm:$false
    Write-Host "[INFO] Tarea anterior '$TaskName1' eliminada"
}

$Action1  = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$BotBat`"" -WorkingDirectory $WorkDir
$Trigger1 = New-ScheduledTaskTrigger -AtLogOn                  # arranca al iniciar sesión
$Settings1 = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0) `               # sin límite de tiempo
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName $TaskName1 `
    -Action   $Action1 `
    -Trigger  $Trigger1 `
    -Settings $Settings1 `
    -RunLevel Highest `
    -Description "TitanBrain: bot de Telegram permanente. Procesa confirmaciones de compra/venta." | Out-Null

Write-Host "[OK] Tarea '$TaskName1' creada — arranca con Windows" -ForegroundColor Green


# ── TAREA 2: Scan horario (lunes-viernes, 7AM-8PM, cada hora) ─────────
$TaskName2 = "TitanBrain_ScanHorario"
if (Get-ScheduledTask -TaskName $TaskName2 -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName2 -Confirm:$false
    Write-Host "[INFO] Tarea anterior '$TaskName2' eliminada"
}

$Action2  = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$ScanBat`"" -WorkingDirectory $WorkDir

# Trigger: cada hora, lunes a viernes, entre 7:00 y 20:00
$Trigger2 = New-ScheduledTaskTrigger `
    -RepetitionInterval (New-TimeSpan -Hours 1) `
    -RepetitionDuration (New-TimeSpan -Hours 13) `              # duración 13h (7AM-8PM)
    -At "07:00" `
    -Daily

$Settings2 = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) `               # máximo 1 hora por ejecución
    -MultipleInstances Queue `
    -StartWhenAvailable                                         # si se perdió una ejecución, corre ni bien pueda

Register-ScheduledTask `
    -TaskName  $TaskName2 `
    -Action    $Action2 `
    -Trigger   $Trigger2 `
    -Settings  $Settings2 `
    -RunLevel  Highest `
    -Description "TitanBrain: scan horario de señales ASX. Lunes-Viernes cada hora." | Out-Null

Write-Host "[OK] Tarea '$TaskName2' creada — cada hora, 7AM-8PM" -ForegroundColor Green


# ── INICIAR AHORA ──────────────────────────────────────────────────────
Write-Host ""
Write-Host "Iniciando tareas ahora..." -ForegroundColor Yellow

# Iniciar bot en background
Start-ScheduledTask -TaskName $TaskName1
Write-Host "[OK] Bot iniciado en background" -ForegroundColor Green

# Correr scan inmediatamente una vez
Write-Host "[OK] Ejecutando primer scan ahora..." -ForegroundColor Green
Start-ScheduledTask -TaskName $TaskName2

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Configuracion completada." -ForegroundColor Cyan
Write-Host ""
Write-Host "  Para verificar estado de tareas:" -ForegroundColor White
Write-Host "    Get-ScheduledTask TitanBrain_Bot | Select State" -ForegroundColor Gray
Write-Host "    Get-ScheduledTask TitanBrain_ScanHorario | Select State" -ForegroundColor Gray
Write-Host ""
Write-Host "  Para ver logs en tiempo real:" -ForegroundColor White
Write-Host "    Get-Content logs\bot_log.txt -Wait" -ForegroundColor Gray
Write-Host "    Get-Content logs\scan_log.txt -Wait" -ForegroundColor Gray
Write-Host ""
Write-Host "  Para detener todo:" -ForegroundColor White
Write-Host "    Stop-ScheduledTask TitanBrain_Bot" -ForegroundColor Gray
Write-Host "    Unregister-ScheduledTask TitanBrain_Bot -Confirm:`$false" -ForegroundColor Gray
Write-Host "    Unregister-ScheduledTask TitanBrain_ScanHorario -Confirm:`$false" -ForegroundColor Gray
Write-Host "==================================================" -ForegroundColor Cyan
