param([Parameter(Mandatory=$true)][string]$DeckPath, [Parameter(Mandatory=$true)][string]$RenderDirectory)
$ErrorActionPreference = 'Stop'
$deckFile = [IO.Path]::GetFullPath($DeckPath)
$renderRoot = [IO.Path]::GetFullPath($RenderDirectory)
[IO.Directory]::CreateDirectory($renderRoot) | Out-Null
$powerPoint = $null
$presentation = $null
try {
    $powerPoint = New-Object -ComObject PowerPoint.Application
    # Read only, no untitled copy, no visible presentation window.
    $presentation = $powerPoint.Presentations.Open($deckFile, -1, 0, 0)
    for ($slideIndex=1; $slideIndex -le $presentation.Slides.Count; $slideIndex++) {
        $imageFile = Join-Path $renderRoot ('slide-{0:D2}.png' -f $slideIndex)
        $presentation.Slides.Item($slideIndex).Export($imageFile, 'PNG', 1280, 720)
    }
    Write-Output ('PowerPoint rendered {0} slides with native significance labels.' -f $presentation.Slides.Count)
}
finally {
    if ($null -ne $presentation) { $presentation.Close(); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($presentation) }
    if ($null -ne $powerPoint) {
        if ($powerPoint.Presentations.Count -eq 0) { $powerPoint.Quit() }
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($powerPoint)
    }
}
