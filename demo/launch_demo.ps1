param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$runtimeDir = Join-Path $PSScriptRoot '.runtime'
[IO.Directory]::CreateDirectory($runtimeDir) | Out-Null
$demoPort = 8511
$demoUrl = "http://127.0.0.1:$demoPort"
$pidFile = Join-Path $runtimeDir 'server.pid'
$demoApp = Join-Path $PSScriptRoot 'app.py'
$serverReady = $false
if (Test-Path -LiteralPath $pidFile) {
    $savedId = [int](Get-Content -LiteralPath $pidFile)
    $savedProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $savedId"
    if ($savedProcess -and $savedProcess.CommandLine.Contains($demoApp) -and $savedProcess.CommandLine.Contains('8511')) {
        try { $serverReady = (Invoke-WebRequest "$demoUrl/_stcore/health" -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200 } catch {}
    }
}
if (-not $serverReady) {
    $candidates = @((Join-Path $repoRoot '.venv\Scripts\python.exe'), 'C:\Python313\python.exe')
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) { $candidates += $pythonCommand.Source }
    $demoPython = $null
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (Test-Path -LiteralPath $candidate) {
            try {
                & $candidate -c 'import streamlit, torch, numpy, pandas, scipy, matplotlib' 2>$null
                if ($LASTEXITCODE -eq 0) { $demoPython = $candidate; break }
            } catch { continue }
        }
    }
    if (-not $demoPython) { throw 'Python with the project dependencies was not found. See demo/README.md.' }
    $arguments = @('-m', 'streamlit', 'run', ('"' + $demoApp + '"'), '--server.address=127.0.0.1', '--server.port=8511', '--server.headless=true', '--browser.gatherUsageStats=false')
    $server = Start-Process -FilePath $demoPython -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtimeDir 'server.stdout.log') -RedirectStandardError (Join-Path $runtimeDir 'server.stderr.log')
    Set-Content -LiteralPath $pidFile -Value $server.Id
    for ($attempt=0; $attempt -lt 40; $attempt++) {
        $server.Refresh()
        if ($server.HasExited) { throw "Demo server stopped. Read $runtimeDir\server.stderr.log" }
        try { $serverReady = (Invoke-WebRequest "$demoUrl/_stcore/health" -UseBasicParsing -TimeoutSec 1).StatusCode -eq 200 } catch {}
        if ($serverReady) { break }
        Start-Sleep -Milliseconds 300
    }
    if (-not $serverReady) { throw "The server is still starting. Logs: $runtimeDir" }
}
if (-not $NoBrowser) {
    $chromePaths = @(
        (Join-Path $env:ProgramFiles 'Google\Chrome\Application\chrome.exe'),
        (Join-Path ${env:ProgramFiles(x86)} 'Google\Chrome\Application\chrome.exe'),
        (Join-Path $env:LOCALAPPDATA 'Google\Chrome\Application\chrome.exe')
    )
    $chromeCommand = Get-Command chrome -ErrorAction SilentlyContinue
    if ($chromeCommand) { $chromePaths += $chromeCommand.Source }
    $chrome = $chromePaths | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $chrome) { throw "Chrome was not found. The demo is running at $demoUrl; open that address in Chrome." }
    Start-Process -FilePath $chrome -ArgumentList @('--new-window', $demoUrl)
}
Write-Output "Demo ready at $demoUrl. Double-click STOP_DEMO.cmd when finished."
