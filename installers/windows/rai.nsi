Unicode true

; RAI for Windows - Setup.
;
; Per user, no administrator rights: RAI goes into %LOCALAPPDATA%\Programs\RAI,
; the folder install.ps1 uses too, with RAI Studio in the Start menu and on the
; desktop, and the folder on the user's PATH so rai, rai-studio and
; rai-get-model work in any new terminal. Models live outside it, in
; %USERPROFILE%\RAI, so they survive upgrades and uninstalling.
;
; Built by the release workflow, which passes everything below that has no
; default: makensis /DVERSION=0.2.6 /DPAYLOAD_DIR=... /DINSTALLSIZE=... /DOUTFILE=...

!define APPNAME     "RAI"
!define COMPANYNAME "ClassEve"
!define DESCRIPTION "Run language models on your computer's processor"
!define HELPURL     "https://github.com/Classevelabs/rai#readme"
!define UPDATEURL   "https://github.com/Classevelabs/rai/releases"
!define ABOUTURL    "https://classeve.com/public/rai"
!define UNINSTKEY   "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}"
!ifndef VERSION
!error "VERSION is not set: build through the release workflow, which passes the workspace version"
!endif
!ifndef PAYLOAD_DIR
!error "PAYLOAD_DIR is not set: build through the release workflow, which stages the files"
!endif
!ifndef INSTALLSIZE
!error "INSTALLSIZE is not set: build through the release workflow, which measures the staged files"
!endif
!ifndef OUTFILE
!error "OUTFILE is not set"
!endif

Name "${APPNAME}"
OutFile "${OUTFILE}"
RequestExecutionLevel user
; A file Windows would not let Setup write offered Ignore, which finished the
; install with that file missing.
AllowSkipFiles off
InstallDir "$LOCALAPPDATA\Programs\${APPNAME}"
InstallDirRegKey HKCU "${UNINSTKEY}" "InstallLocation"
SetCompressor /SOLID lzma
BrandingText "${APPNAME} ${VERSION}"

VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName" "${APPNAME}"
VIAddVersionKey "CompanyName" "${COMPANYNAME}"
VIAddVersionKey "FileDescription" "${APPNAME} Setup"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "LegalCopyright" "${COMPANYNAME}"

!include LogicLib.nsh
!include x64.nsh
!include WinMessages.nsh
!include MUI2.nsh

!define MUI_ABORTWARNING
; No folder page: RAI installs for one person, in the folder Windows keeps for
; that, and a typed folder such as D:\Tools would be one the uninstaller later
; had to tell apart from the user's own files.
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_RUN "$INSTDIR\rai-studio.cmd"
!define MUI_FINISHPAGE_RUN_TEXT "Open RAI Studio (it offers to download a model to chat with)"
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

; One Setup or uninstall at a time: a second one started while the first is
; writing (a double-click while SmartScreen is still asking) would clear the
; folder the first is filling.
!macro RAI_SINGLE_SETUP
    System::Call 'kernel32::CreateMutexW(p 0, i 1, w "ClassEve.RAI.Setup") p .r0 ?e'
    Pop $1
    ${If} $1 = 183
        MessageBox MB_OK|MB_ICONEXCLAMATION "RAI Setup is already running. Finish that one first." /SD IDOK
        Abort
    ${EndIf}
!macroend

; The files RAI installs, which an update replaces and the uninstaller
; removes: only these, so anything else in the folder is left alone. The
; release workflow fails the build if the payload holds a file this list does
; not name. rai-convert.exe, rai-generate.exe, rai-chat.exe and RUNNING.txt
; come with install.ps1, which uses the same folder.
!macro RAI_DELETE_FILES
    Delete "$INSTDIR\rai.exe"
    Delete "$INSTDIR\rai-convert.exe"
    Delete "$INSTDIR\rai-generate.exe"
    Delete "$INSTDIR\rai-chat.exe"
    Delete "$INSTDIR\rai-studio.cmd"
    Delete "$INSTDIR\rai-get-model.cmd"
    Delete "$INSTDIR\rai-get-model-windows.ps1"
    Delete "$INSTDIR\models.tsv"
    Delete "$INSTDIR\LICENSE"
    Delete "$INSTDIR\NOTICE"
    Delete "$INSTDIR\THIRD-PARTY-NOTICES.md"
    Delete "$INSTDIR\README.md"
    Delete "$INSTDIR\INSTALL.md"
    Delete "$INSTDIR\RUNNING.txt"
    Delete "$INSTDIR\.rai-install"
!macroend

; PATH is edited through the registry with its %VARIABLES% left unexpanded,
; never through NSIS strings, which stop at 1,024 characters and would write
; a long PATH back cut short. The folder reaches PowerShell in an environment
; variable, so no quote in a folder name can break the command.
!define RAI_PATH_ADD `powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "$$d=$$env:RAI_SETUP_DIR; $$k=[Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment',$$true); $$p=[string]$$k.GetValue('Path','',[Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames); $$e=@($$p.Split(';')|Where-Object{$$_}); if(-not($$e|Where-Object{$$_.TrimEnd('\') -ieq $$d.TrimEnd('\')})){$$k.SetValue('Path',(($$e+$$d) -join ';'),[Microsoft.Win32.RegistryValueKind]::ExpandString)}; $$k.Close()"`
!define RAI_PATH_REMOVE `powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "$$d=$$env:RAI_SETUP_DIR; $$k=[Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment',$$true); $$p=[string]$$k.GetValue('Path','',[Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames); $$e=@($$p.Split(';')|Where-Object{$$_ -and ($$_.TrimEnd('\') -ine $$d.TrimEnd('\'))}); $$k.SetValue('Path',($$e -join ';'),[Microsoft.Win32.RegistryValueKind]::ExpandString); $$k.Close()"`

Function .onInit
    !insertmacro RAI_SINGLE_SETUP
    ${IfNot} ${RunningX64}
        MessageBox MB_OK|MB_ICONSTOP "RAI needs 64-bit Windows." /SD IDOK
        Abort
    ${EndIf}
FunctionEnd

Function un.onInit
    !insertmacro RAI_SINGLE_SETUP
FunctionEnd

Section "Install"
    ; A running rai.exe (RAI Studio left open) cannot be replaced; say so,
    ; instead of failing part-way through the copy.
  check_running:
    ClearErrors
    ${If} ${FileExists} "$INSTDIR\rai.exe"
        Delete "$INSTDIR\rai.exe"
        ${If} ${Errors}
            MessageBox MB_RETRYCANCEL|MB_ICONEXCLAMATION "RAI is running (RAI Studio may be open). Close it, then choose Retry." /SD IDCANCEL IDRETRY check_running
            Abort
        ${EndIf}
    ${EndIf}

    ; An update replaces RAI's own files, so none a newer version dropped
    ; lingers, and leaves anything else in the folder alone. Models are in
    ; %USERPROFILE%\RAI and are not touched.
    !insertmacro RAI_DELETE_FILES

    SetOutPath "$INSTDIR"
    ; rai.exe last: started while Setup is still writing, it never finds the
    ; launchers and model list missing.
    File /r /x "rai.exe" "${PAYLOAD_DIR}\*.*"
    File "${PAYLOAD_DIR}\rai.exe"
    ; The mark install.ps1 looks for before it replaces this folder.
    FileOpen $0 "$INSTDIR\.rai-install" w
    FileWrite $0 "${VERSION}$\r$\n"
    FileClose $0

    CreateDirectory "$SMPROGRAMS\${APPNAME}"
    CreateShortcut "$SMPROGRAMS\${APPNAME}\RAI Studio.lnk" "$INSTDIR\rai-studio.cmd" "" "$INSTDIR\rai.exe" 0
    CreateShortcut "$SMPROGRAMS\${APPNAME}\Uninstall RAI.lnk" "$INSTDIR\uninstall.exe"
    CreateShortcut "$DESKTOP\RAI Studio.lnk" "$INSTDIR\rai-studio.cmd" "" "$INSTDIR\rai.exe" 0
    ; install.ps1 puts its one entry at the top of the Start menu. Setup's own
    ; entries replace it, so the menu never shows RAI twice.
    Delete "$SMPROGRAMS\RAI Studio.lnk"

    System::Call 'kernel32::SetEnvironmentVariable(t "RAI_SETUP_DIR", t "$INSTDIR")'
    nsExec::Exec `${RAI_PATH_ADD}`
    Pop $0
    SendMessage ${HWND_BROADCAST} ${WM_SETTINGCHANGE} 0 "STR:Environment" /TIMEOUT=5000

    WriteUninstaller "$INSTDIR\uninstall.exe"
    WriteRegStr HKCU "${UNINSTKEY}" "DisplayName" "${APPNAME}"
    WriteRegStr HKCU "${UNINSTKEY}" "UninstallString" '"$INSTDIR\uninstall.exe"'
    WriteRegStr HKCU "${UNINSTKEY}" "QuietUninstallString" '"$INSTDIR\uninstall.exe" /S'
    WriteRegStr HKCU "${UNINSTKEY}" "InstallLocation" "$INSTDIR"
    WriteRegStr HKCU "${UNINSTKEY}" "DisplayIcon" "$INSTDIR\rai.exe"
    WriteRegStr HKCU "${UNINSTKEY}" "Publisher" "${COMPANYNAME}"
    WriteRegStr HKCU "${UNINSTKEY}" "Comments" "${DESCRIPTION}"
    WriteRegStr HKCU "${UNINSTKEY}" "HelpLink" "${HELPURL}"
    WriteRegStr HKCU "${UNINSTKEY}" "URLUpdateInfo" "${UPDATEURL}"
    WriteRegStr HKCU "${UNINSTKEY}" "URLInfoAbout" "${ABOUTURL}"
    WriteRegStr HKCU "${UNINSTKEY}" "DisplayVersion" "${VERSION}"
    WriteRegDWORD HKCU "${UNINSTKEY}" "NoModify" 1
    WriteRegDWORD HKCU "${UNINSTKEY}" "NoRepair" 1
    WriteRegDWORD HKCU "${UNINSTKEY}" "EstimatedSize" ${INSTALLSIZE}
SectionEnd

Section "Uninstall"
    ; A running rai.exe (RAI Studio left open) cannot be removed. Say so, rather
    ; than removing the Apps entry and leaving the program behind.
  un_check_running:
    ClearErrors
    ${If} ${FileExists} "$INSTDIR\rai.exe"
        Delete "$INSTDIR\rai.exe"
        ${If} ${Errors}
            MessageBox MB_RETRYCANCEL|MB_ICONEXCLAMATION "RAI is running (RAI Studio may be open). Close it, then choose Retry." /SD IDCANCEL IDRETRY un_check_running
            Abort
        ${EndIf}
    ${EndIf}

    Delete "$SMPROGRAMS\${APPNAME}\RAI Studio.lnk"
    Delete "$SMPROGRAMS\${APPNAME}\Uninstall RAI.lnk"
    RMDir "$SMPROGRAMS\${APPNAME}"
    Delete "$DESKTOP\RAI Studio.lnk"
    Delete "$SMPROGRAMS\RAI Studio.lnk"

    System::Call 'kernel32::SetEnvironmentVariable(t "RAI_SETUP_DIR", t "$INSTDIR")'
    nsExec::Exec `${RAI_PATH_REMOVE}`
    Pop $0
    SendMessage ${HWND_BROADCAST} ${WM_SETTINGCHANGE} 0 "STR:Environment" /TIMEOUT=5000

    ; Models are big and slow to download again, and uninstall-then-reinstall
    ; is the first thing people try when something is wrong, so No is the
    ; default and the answer a silent uninstall gives.
    ${If} ${FileExists} "$PROFILE\RAI\models\*.*"
        MessageBox MB_YESNO|MB_ICONQUESTION|MB_DEFBUTTON2 "Also delete your downloaded models in $PROFILE\RAI?$\r$\nChoose No to keep them for a future reinstall." /SD IDNO IDNO keep_models
        RMDir /r "$PROFILE\RAI"
      keep_models:
    ${EndIf}

    ; RAI's files only, then the folder if nothing else is in it: a folder the
    ; user also keeps other things in keeps them.
    !insertmacro RAI_DELETE_FILES
    Delete "$INSTDIR\uninstall.exe"
    RMDir "$INSTDIR"
    DeleteRegKey HKCU "${UNINSTKEY}"
SectionEnd
