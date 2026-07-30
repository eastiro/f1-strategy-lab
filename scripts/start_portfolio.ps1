param(
    [int]$Port = 8507
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Streamlit = "C:\Users\EASTIRO\AppData\Local\Programs\Python\Python313\Scripts\streamlit.exe"

Set-Location $ProjectRoot
& $Streamlit run app/portfolio_app.py `
    --server.port $Port `
    --server.headless true `
    --server.fileWatcherType none `
    --browser.gatherUsageStats false
