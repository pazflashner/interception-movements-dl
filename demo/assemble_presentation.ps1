param([string]$SourcePath, [string]$NewSlidesPath)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $SourcePath) { $SourcePath = Join-Path $repoRoot '.tmp\demo_build\Paz_latest.pptx' }
if (-not $NewSlidesPath) { $NewSlidesPath = Join-Path $PSScriptRoot '.build\new_slides.pptx' }
$workingCopy = Join-Path $PSScriptRoot '.build\combined_work.pptx'
$outputPath = Join-Path $PSScriptRoot 'Presentation_Demo.pptx'
$pdfPath = Join-Path $PSScriptRoot 'Presentation_Demo.pdf'
Copy-Item -LiteralPath $SourcePath -Destination $workingCopy -Force
$powerPoint = $null
$deck = $null
try {
    $powerPoint = New-Object -ComObject PowerPoint.Application
    $deck = $powerPoint.Presentations.Open($workingCopy, 0, 0, 0)
    for ($i=$deck.Slides.Count; $i -gt 8; $i--) { $deck.Slides.Item($i).Delete() }
    [void]$deck.Slides.InsertFromFile($NewSlidesPath, 8, 1, 4)
    # InsertFromFile can inherit the destination master's white background.
    $closing = $deck.Slides.Item(12)
    $closing.FollowMasterBackground = 0
    $closing.Background.Fill.Solid()
    $closing.Background.Fill.ForeColor.RGB = 14 + 23 * 256 + 41 * 65536
    # Only the title slide is edited within Paz's first eight, per the user's follow-up.
    foreach ($shape in $deck.Slides.Item(1).Shapes) {
        if ($shape.HasTextFrame -and $shape.TextFrame.HasText) {
            $text = $shape.TextFrame.TextRange.Text
            if ($text -like 'Research supervision:*') {
                $shape.TextFrame.TextRange.Text = "Research supervisor: Prof. Jason Friedman   |   Course advisor: Shimon Shahar`rCourse lecturer: Prof. Nadav Cohen   |   Teaching assistant: Yonatan Slutzky"
                $shape.Top = 353
                $shape.Height = 47
                $shape.TextFrame.TextRange.Font.Size = 12
            } elseif ($text -like 'Workshop on Deep Learning*') {
                $shape.TextFrame.TextRange.Text = 'Workshop on Deep Learning  |  03683538  |  Tel Aviv University'
                $shape.Top = 401
                $shape.Height = 22
                $shape.TextFrame.TextRange.Font.Size = 12
            } elseif ($text -like 'Planned talk:*') {
                $shape.TextFrame.TextRange.Text = '10-minute presentation and live demo'
            }
        }
    }
    $deck.SaveAs($outputPath, 24)
    $deck.SaveAs($pdfPath, 32)
    Write-Output "Created $($deck.Slides.Count) slides: $outputPath and PDF"
} finally {
    if ($null -ne $deck) { $deck.Close(); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($deck) }
    if ($null -ne $powerPoint) {
        if ($powerPoint.Presentations.Count -eq 0) { $powerPoint.Quit() }
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($powerPoint)
    }
}
