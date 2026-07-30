$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Set-Location $ProjectRoot

python -m compileall -q app src
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

python -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$appCheck = @'
from streamlit.testing.v1 import AppTest

for app_path in ("app/streamlit_app.py", "app/portfolio_app.py"):
    app = AppTest.from_file(app_path)
    app.run(timeout=45)
    assert not app.exception, [error.value for error in app.exception]

print("Project checks passed")
'@

$appCheck | python -B -
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
