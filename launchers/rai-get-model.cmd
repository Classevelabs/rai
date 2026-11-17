@echo off
rem rai-get-model.cmd - download a model from Hugging Face and convert it.
rem Runs rai-get-model.ps1 beside it with the same arguments; see that file,
rem or run: rai-get-model --help
rem Started from a PowerShell 7 window, Windows PowerShell would inherit that
rem window's module path and lose cmdlets of its own; it rebuilds its own when
rem the variable is unset.
setlocal
set "PSModulePath="
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0rai-get-model.ps1" %*
exit /b %ERRORLEVEL%
