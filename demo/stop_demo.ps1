$ErrorActionPreference = 'Stop'
$pidFile = Join-Path $PSScriptRoot '.runtime\server.pid'
if (Test-Path -LiteralPath $pidFile) {
    $demoProcess = Get-CimInstance Win32_Process -Filter ("ProcessId = " + [int](Get-Content -LiteralPath $pidFile))
    $demoApp = Join-Path $PSScriptRoot 'app.py'
    if ($demoProcess -and $demoProcess.CommandLine.Contains($demoApp) -and $demoProcess.CommandLine.Contains('8511')) {
        Stop-Process -Id $demoProcess.ProcessId
        Write-Output 'Demo server stopped.'
    } elseif ($demoProcess) { throw 'The saved process belongs to another program; it was not stopped.' }
    Remove-Item -LiteralPath $pidFile
}
