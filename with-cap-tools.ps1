$script = Join-Path $PSScriptRoot "with-cap-tools-pip.ps1"
& $script @args
exit $LASTEXITCODE
