if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtualenv .venv..."
    python -m venv .venv
}
Write-Host "Activate it with: .\\.venv\\Scripts\\Activate.ps1"
if (Test-Path "requirements.txt") {
    Write-Host "Installing requirements (fast check)..."
    python -m pip install -r requirements.txt
} else {
    Write-Host "No requirements.txt found; install manually from README_AU_STOCK.md"
}
Write-Host "Quick check: import core libs"
$py = @"
try:
    import pandas, numpy
    print('DeepQuant: core libs available')
except Exception as e:
    print('DeepQuant check failed:', e)
"@
$pyPath = Join-Path $PSScriptRoot 'deepquant_check.py'
Set-Content -Path $pyPath -Value $py -Encoding UTF8
python $pyPath
Remove-Item $pyPath -Force
