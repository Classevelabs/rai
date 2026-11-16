@echo off
rem ---------------------------------------------------------------------------
rem rai-studio.cmd - double-clickable launcher for the local rai chat UI.
rem
rem Put this file in the same folder as rai.exe and your .raimodel file, then
rem double-click it. It starts `rai serve`, waits for the port to come up, and
rem opens your default browser. Closing this window stops the server.
rem
rem Override the defaults with environment variables:
rem   set RAI_PORT=9000
rem   set RAI_MODEL=C:\models\tinyllama-1.1b-q4.raimodel
rem   set RAI_NO_BROWSER=1     start the server without opening a browser
rem ---------------------------------------------------------------------------
rem Delayed expansion stays off: `!` is legal in a Windows path, and with it on
rem every path holding one lost it. Paths are expanded only inside double
rem quotes and never inside a parenthesized block, because a bare `&` runs a
rem command and a bare `)` ends the block, as in "rai (1)".
setlocal disabledelayedexpansion

set "HERE=%~dp0"
if not defined RAI_PORT set "RAI_PORT=8090"

rem --- where this launcher looks for things ----------------------------------
rem The release archive ships rai.exe at its root and the launchers one level
rem down, so the parent directory is searched too. Looking only beside this
rem script is what made a downloaded archive fail on the first double-click.
set "ROOT=%HERE%"
if exist "%HERE%..\rai.exe" for %%D in ("%HERE%..") do set "ROOT=%%~fD\"

rem --- locate rai: beside this script, then the archive root, then PATH -----
set "RAI=%HERE%rai.exe"
if not exist "%RAI%" set "RAI=%ROOT%rai.exe"
if exist "%RAI%" goto :pick_model
set "RAI="
for %%I in (rai.exe) do set "RAI=%%~$PATH:I"
if defined RAI goto :pick_model
echo.
echo   Could not find rai.exe. Put it in this folder or on your PATH:
echo     "%ROOT%"
echo   then run this launcher again.
echo.
pause
exit /b 1

rem --- pick a model ----------------------------------------------------------
:pick_model
if not defined RAI_MODEL goto :find_model
set "MODEL=%RAI_MODEL%"
if exist "%MODEL%" goto :start
echo.
echo   RAI_MODEL is set to a file that does not exist:
echo     "%MODEL%"
echo.
pause
exit /b 1

:find_model
set "MODEL="
set "COUNT=0"
for %%F in ("%ROOT%*.raimodel") do set /a COUNT+=1 & set "MODEL=%%~fF"
if "%COUNT%"=="1" goto :start
if not "%COUNT%"=="0" goto :many_models
echo.
echo   No .raimodel file found in this folder:
echo     "%ROOT%"
echo.
echo   Convert a HuggingFace checkpoint first, for example:
echo     "%RAI%" convert C:\path\to\TinyLlama-1.1B-Chat -o "%ROOT%tinyllama.raimodel"
echo.
echo   Then run this launcher again.
echo.
pause
exit /b 1

:many_models
echo.
echo   More than one .raimodel file is in this folder, so it is not
echo   obvious which one to start:
echo.
for %%F in ("%ROOT%*.raimodel") do echo     %%~nxF
echo.
echo   Pick one by setting RAI_MODEL and running this launcher again:
echo     set "RAI_MODEL=%ROOT%<name>.raimodel"
echo.
pause
exit /b 1

:start
echo   Model:  "%MODEL%"
echo   Server: http://localhost:%RAI_PORT%
echo.
echo   Waiting for the model to load, then opening your browser.
echo   Close this window to stop the server.
echo.

rem --- open the browser once the port accepts a connection --------------------
rem Runs alongside the server below, in this same console. `localhost` rather
rem than 127.0.0.1, because rai serve binds IPv6 loopback on a machine without
rem an IPv4 one. A large model can take longer to load than this waits, so the
rem message says what to do rather than declaring a failure.
if defined RAI_NO_BROWSER goto :serve
start "" /b powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$deadline = (Get-Date).AddSeconds(120);" ^
  "while ((Get-Date) -lt $deadline) {" ^
  "  try { $c = New-Object Net.Sockets.TcpClient; $c.Connect('localhost', %RAI_PORT%); $c.Close();" ^
  "        Start-Process 'http://localhost:%RAI_PORT%/'; exit 0 }" ^
  "  catch { Start-Sleep -Milliseconds 500 } };" ^
  "Write-Host '  Still not answering after two minutes. If the model is still loading,';" ^
  "Write-Host '  open http://localhost:%RAI_PORT%/ yourself once it is ready.'"

rem --- run the server in this window so closing the window stops it ----------
:serve
"%RAI%" serve "%MODEL%" --port %RAI_PORT%
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" exit /b 0
echo.
echo   rai serve exited with code %RC%.
pause
exit /b %RC%
