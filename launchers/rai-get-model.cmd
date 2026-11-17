@echo off
rem rai-get-model.cmd - download a model from Hugging Face and convert it.
rem Runs rai-get-model.ps1 beside it with the same arguments; see that file,
rem or run: rai-get-model --help
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0rai-get-model.ps1" %*
exit /b %ERRORLEVEL%
