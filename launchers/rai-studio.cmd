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
rem ---------------------------------------------------------------------------
setlocal enabledelayedexpansion

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
if not exist "%RAI%" (
    set "RAI="
    for %%I in (rai.exe) do set "RAI=%%~$PATH:I"
)
if not defined RAI (
    echo.
    echo   Could not find rai.exe.
    echo.
    echo   Put rai.exe in this folder ^(%ROOT%^) or on your PATH,
    echo   then run this launcher again.
    echo.
    pause
    exit /b 1
)

rem --- pick a model ----------------------------------------------------------
if defined RAI_MODEL (
    if not exist "%RAI_MODEL%" (
        echo.
        echo   RAI_MODEL is set to "%RAI_MODEL%" but that file does not exist.
        echo.
        pause
        exit /b 1
    )
    set "MODEL=%RAI_MODEL%"
) else (
    set "MODEL="
    set /a COUNT=0
    for %%F in ("%ROOT%*.raimodel") do (
        set /a COUNT+=1
        set "MODEL=%%~fF"
    )

    if !COUNT! EQU 0 (
        echo.
        echo   No .raimodel file found in this folder:
        echo     %ROOT%
        echo.
        echo   Convert a HuggingFace checkpoint first, for example:
        echo     "%RAI%" convert C:\path\to\TinyLlama-1.1B-Chat -o "%ROOT%tinyllama.raimodel"
        echo.
        echo   Then run this launcher again.
        echo.
        pause
        exit /b 1
    )

    if !COUNT! GTR 1 (
        echo.
        echo   More than one .raimodel file is in this folder, so it is not
        echo   obvious which one to start:
        echo.
        for %%F in ("%ROOT%*.raimodel") do echo     %%~nxF
        echo.
        echo   Pick one by setting RAI_MODEL and running this launcher again:
        echo     set RAI_MODEL=%ROOT%^<name^>.raimodel
        echo.
        pause
        exit /b 1
    )
)

echo   Model:  %MODEL%
echo   Server: http://localhost:%RAI_PORT%
echo.
echo   Waiting for the model to load, then opening your browser.
echo   Close this window to stop the server.
echo.

rem --- open the browser once the port accepts a connection --------------------
rem Runs alongside the server below; gives up after two minutes rather than
rem opening a tab at a URL that will never answer.
start "" /b powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$deadline = (Get-Date).AddSeconds(120);" ^
  "while ((Get-Date) -lt $deadline) {" ^
  "  try { $c = New-Object Net.Sockets.TcpClient; $c.Connect('127.0.0.1', %RAI_PORT%); $c.Close();" ^
  "        Start-Process 'http://localhost:%RAI_PORT%/'; exit 0 }" ^
  "  catch { Start-Sleep -Milliseconds 500 } };" ^
  "Write-Host '  The server did not start; see the messages above.'"

rem --- run the server in this window so closing the window stops it ----------
"%RAI%" serve "%MODEL%" --port %RAI_PORT%
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo   rai serve exited with code %RC%.
    pause
)
exit /b %RC%
