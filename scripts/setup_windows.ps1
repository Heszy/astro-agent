$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$pythonCommand = $null
$pythonArgs = @()

if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.12 -c "import sys; print(sys.executable)" *> $null
    if ($LASTEXITCODE -eq 0) {
        $pythonCommand = "py"
        $pythonArgs = @("-3.12")
    }
}

if (-not $pythonCommand) {
    $localPython = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
    if (Test-Path -LiteralPath $localPython) {
        $pythonCommand = $localPython
    }
}

if (-not $pythonCommand -and (Get-Command python -ErrorAction SilentlyContinue)) {
    & python -c "import sys; print(sys.executable)" *> $null
    if ($LASTEXITCODE -eq 0) {
        $pythonCommand = "python"
    }
}

if (-not $pythonCommand) {
    throw "Python 3.12 was not found. Install it first: winget install Python.Python.3.12"
}

Write-Host "Using Python: $pythonCommand $($pythonArgs -join ' ')"
& $pythonCommand @pythonArgs -m venv .venv
if ($LASTEXITCODE -ne 0) {
    throw "Python failed to create the virtual environment"
}

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Virtual environment creation failed: $venvPython not found"
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -e ".[dev]"
& $venvPython scripts\generate_sample_data.py
& $venvPython -m pytest -q

$envFile = Join-Path $projectRoot ".env"
if (-not (Test-Path -LiteralPath $envFile)) {
    Copy-Item -LiteralPath (Join-Path $projectRoot ".env.example") -Destination $envFile
    Write-Host "Created .env from .env.example"
}
else {
    Write-Host ".env already exists; it was not overwritten"
}

Write-Host ""
Write-Host "Windows environment and tests are ready. Next:"
Write-Host "  1. Open .env and replace DEEPSEEK_API_KEY with your real key"
Write-Host "  2. Run: .\.venv\Scripts\Activate.ps1"
Write-Host "  3. Run: uvicorn app.api:app --reload"
