@echo off
rem ---------------------------------------------------------------------------
rem rai-studio.cmd - double-click to chat with a model in your browser.
rem
rem It finds rai.exe and a model, starts `rai serve`, and opens your browser at
rem http://localhost:8090. With no model on this computer yet, it offers to
rem download the recommended one, and starts the page either way. Closing this
rem window stops the server.
rem
rem Settings, as environment variables:
rem   set RAI_PORT=9000                       another port
rem   set RAI_MODEL=C:\models\my.raimodel     the model to start with
rem   set RAI_HOME=D:\RAI                     where models live (%USERPROFILE%\RAI)
rem   set RAI_NO_BROWSER=1                    do not open a browser
rem ---------------------------------------------------------------------------
rem Delayed expansion stays off: `!` is legal in a Windows path, and with it on
rem every path holding one lost it. Paths are expanded only inside double
rem quotes and never inside a parenthesized block, because a bare `&` runs a
rem command and a bare `)` ends the block, as in "rai (1)".
setlocal disabledelayedexpansion

set "HERE=%~dp0"
if not defined RAI_PORT set "RAI_PORT=8090"
if not defined RAI_HOME set "RAI_HOME=%USERPROFILE%\RAI"

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

rem --- pick a model: RAI_MODEL, then beside rai.exe, then %RAI_HOME% ----------
:pick_model
set "MODEL="
if not defined RAI_MODEL goto :find_beside
set "MODEL=%RAI_MODEL%"
if exist "%MODEL%" goto :start
echo.
echo   RAI_MODEL is set to a file that does not exist:
echo     "%MODEL%"
echo.
pause
exit /b 1

rem One model next to rai.exe, as an unpacked archive keeps it, starts at
rem once. Several start the page with none loaded, so you choose there.
:find_beside
set "COUNT=0"
for %%F in ("%ROOT%*.raimodel") do set /a COUNT+=1 & set "MODEL=%%~fF"
if "%COUNT%"=="1" goto :start
set "MODEL="
if not "%COUNT%"=="0" goto :start

rem The models rai-get-model converted, one folder each under %RAI_HOME%\models.
:find_home
set "COUNT=0"
if exist "%RAI_HOME%\models\" for /r "%RAI_HOME%\models" %%F in (*.raimodel) do set /a COUNT+=1 & set "MODEL=%%~fF"
if "%COUNT%"=="1" goto :start
set "MODEL="
if not "%COUNT%"=="0" goto :start

rem --- no model yet: offer the recommended one -----------------------------
rem With no console to type into, `choice` fails, and that counts as No: a
rem launcher started that way never waits for an answer.
set "GET=%HERE%rai-get-model.cmd"
if not exist "%GET%" goto :start
echo.
echo   RAI needs a model to chat with, and there is none on this computer yet.
echo   The recommended one is Qwen2.5 1.5B Instruct, a 3.1 GB download. It is
echo   saved in "%RAI_HOME%" and ready to chat as soon as it is converted.
echo.
choice /c YN /n /m "  Download it now? [Y/N] "
if errorlevel 2 goto :start
call "%GET%" qwen2.5-1.5b --yes
if errorlevel 1 goto :start
set "MODEL=%RAI_HOME%\models\qwen2.5-1.5b\qwen2.5-1.5b-q4.raimodel"
if not exist "%MODEL%" set "MODEL="

:start
echo.
if defined MODEL echo   Model:  "%MODEL%"
if not defined MODEL echo   Model:  none yet - choose or get one in the page that opens
echo   Server: http://localhost:%RAI_PORT%
echo.
echo   Waiting for the server, then opening your browser.
echo   Close this window to stop the server.
echo.

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
rem From the folder rai.exe is in, so the page's library lists models kept
rem beside it as well as those under %RAI_HOME%.
:serve
cd /d "%ROOT%"
if defined MODEL "%RAI%" serve "%MODEL%" --port %RAI_PORT%
if not defined MODEL "%RAI%" serve --port %RAI_PORT%
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" exit /b 0
echo.
echo   rai serve exited with code %RC%.
pause
exit /b %RC%
