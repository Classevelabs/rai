@echo off
rem rai-get-model.cmd - download a model from Hugging Face and convert it.
rem Runs rai-get-model-windows.ps1 beside it with the same arguments; see that
rem file, or run: rai-get-model --help. That file has a name of its own because
rem PowerShell runs a .ps1 before a .cmd of the same name, and Windows' default
rem policy refuses scripts.
rem Started from a PowerShell 7 window, Windows PowerShell would inherit that
rem window's module path and lose cmdlets of its own; it rebuilds its own when
rem the variable is unset.
setlocal
set "PSModulePath="
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0rai-get-model-windows.ps1" %*
exit /b %ERRORLEVEL%
