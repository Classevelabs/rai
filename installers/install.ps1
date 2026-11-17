# Install RAI on Windows. In PowerShell:
#
#   irm https://github.com/Classevelabs/rai/releases/latest/download/install.ps1 | iex
#
# It downloads the RAI release, checks it against the release's SHA256SUMS,
# installs it for you alone (no administrator rights) into
# %LOCALAPPDATA%\Programs\RAI, the folder RAI's Setup.exe uses too, adds it to
# your PATH and the Start menu, and offers to download a model to chat with.
#
# Settings, as environment variables set before running it:
#   $env:RAI_INSTALL_MODEL = 'qwen2.5-0.5b'   also download this starter model
#   $env:RAI_NO_MODEL = '1'                   do not offer a model
#   $env:RAI_VERSION = '0.2.6'                install this release, not the latest
#   $env:RAI_INSTALL_DIR = 'D:\Tools\RAI'     install somewhere else
#   $env:RAI_NO_PATH = '1'                    leave PATH alone
#
# Written for Windows PowerShell 5.1, the version every Windows 10 and 11 has.

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$RepoUrl = 'https://github.com/Classevelabs/rai'
$Target = 'x86_64-pc-windows-msvc'
$Version = if ($env:RAI_VERSION) { $env:RAI_VERSION.TrimStart('v') } else { '' }
$Dir = if ($env:RAI_INSTALL_DIR) { $env:RAI_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA 'Programs\RAI' }
# A folder that already holds the release files, used instead of the internet.
# This is how the release is tested on clean machines before it is published.
$From = $env:RAI_INSTALL_FROM
$Model = $env:RAI_INSTALL_MODEL

function Say([string]$text) { Write-Host $text }
function Fail([string]$text) {
    Write-Host ''
    Write-Host "RAI installer: $text" -ForegroundColor Red
    Write-Host ''
    # `exit` inside `irm | iex` would close the user's PowerShell window.
    throw 'RAI was not installed.'
}

if (-not [Environment]::Is64BitOperatingSystem) { Fail 'RAI needs 64-bit Windows.' }

# --- which release ------------------------------------------------------------------
if (-not $Version) {
    if ($From) {
        $found = Get-ChildItem -LiteralPath $From -Filter "rai-*-$Target.zip" | Select-Object -First 1
        if ($found) { $Version = $found.Name -replace "^rai-(.*)-$([regex]::Escape($Target))\.zip$", '$1' }
    } else {
        # GitHub answers /releases/latest with a redirect to the release's tag.
        try {
            $request = [System.Net.HttpWebRequest]::Create("$RepoUrl/releases/latest")
            $request.AllowAutoRedirect = $false
            $response = $request.GetResponse()
            $location = $response.Headers['Location']
            $response.Close()
            if ($location -match '/tag/v([^/]+)$') { $Version = $Matches[1] }
        } catch { }
    }
}
if ($Version -notmatch '^\d+\.\d+\.\d+') { Fail 'could not find out the latest RAI version. Set $env:RAI_VERSION, for example to 0.2.6, and run this again.' }

$Archive = "rai-$Version-$Target.zip"
$Temp = Join-Path ([System.IO.Path]::GetTempPath()) ("rai-install-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $Temp | Out-Null

try {
    Say ''
    Say "Installing RAI $Version"
    if ($From) {
        Copy-Item -LiteralPath (Join-Path $From $Archive), (Join-Path $From 'SHA256SUMS') -Destination $Temp
    } else {
        $base = "$RepoUrl/releases/download/v$Version"
        foreach ($name in $Archive, 'SHA256SUMS') {
            try { Invoke-WebRequest -Uri "$base/$name" -OutFile (Join-Path $Temp $name) -UseBasicParsing }
            catch { Fail "could not download $base/$name ($($_.Exception.Message))" }
        }
    }

    $expected = $null
    foreach ($line in Get-Content -LiteralPath (Join-Path $Temp 'SHA256SUMS')) {
        $parts = $line -split '\s+', 2
        if ($parts.Count -eq 2 -and $parts[1].TrimStart('*') -eq $Archive) { $expected = $parts[0].ToLowerInvariant() }
    }
    if (-not $expected) { Fail "SHA256SUMS has no entry for $Archive." }
    $actual = (Get-FileHash -LiteralPath (Join-Path $Temp $Archive) -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) { Fail "the download does not match SHA256SUMS (got $actual, expected $expected). Nothing was installed." }
    Say 'Checked: SHA-256 matches the release.'

    Expand-Archive -LiteralPath (Join-Path $Temp $Archive) -DestinationPath $Temp -Force
    $unpacked = Join-Path $Temp "rai-$Version-$Target"
    if (-not (Test-Path -LiteralPath (Join-Path $unpacked 'rai.exe'))) { Fail 'the archive has no rai.exe in it.' }

    # A running rai.exe (RAI Studio left open) cannot be replaced.
    $running = Get-Process -Name rai -ErrorAction SilentlyContinue | Where-Object {
        try { $_.Path -and $_.Path.StartsWith($Dir, [StringComparison]::OrdinalIgnoreCase) } catch { $false }
    }
    if ($running) { Fail 'RAI is running (RAI Studio may be open). Close it, then run this again.' }

    # The program, then the launchers beside it in one folder, so one PATH
    # entry covers rai, rai-studio and rai-get-model.
    # Only a folder that is empty or holds a previous RAI is ever replaced:
    # RAI_INSTALL_DIR pointed at a folder of other things must not delete them.
    # A folder RAI's Setup made keeps Setup's uninstaller, which Settings >
    # Apps runs, and Setup's own Start menu entry, so updating it this way
    # leaves one entry and an uninstaller that still removes everything.
    $setupUninstaller = $null
    if (Test-Path -LiteralPath $Dir) {
        $earlier = (Test-Path -LiteralPath (Join-Path $Dir 'rai.exe')) -or (Test-Path -LiteralPath (Join-Path $Dir 'uninstall.exe'))
        if (-not $earlier -and (Get-ChildItem -LiteralPath $Dir -Force | Select-Object -First 1)) {
            Fail "$Dir already holds other files and is not an earlier RAI install. Choose an empty folder with RAI_INSTALL_DIR."
        }
        if (Test-Path -LiteralPath (Join-Path $Dir 'uninstall.exe')) {
            $setupUninstaller = Join-Path $Temp 'uninstall.exe'
            Copy-Item -LiteralPath (Join-Path $Dir 'uninstall.exe') -Destination $setupUninstaller
        }
        Remove-Item -LiteralPath $Dir -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $Dir | Out-Null
    if ($setupUninstaller) { Copy-Item -LiteralPath $setupUninstaller -Destination $Dir }
    Get-ChildItem -LiteralPath $unpacked -File | Where-Object { $_.Name -ne 'rai.exe' } |
        Copy-Item -Destination $Dir
    foreach ($name in 'rai-studio.cmd', 'rai-get-model.cmd', 'rai-get-model.ps1', 'models.tsv') {
        Copy-Item -LiteralPath (Join-Path (Join-Path $unpacked 'launchers') $name) -Destination $Dir
    }
    # rai.exe last: a copy started while this runs never finds its launchers missing.
    Copy-Item -LiteralPath (Join-Path $unpacked 'rai.exe') -Destination $Dir
    $rai = Join-Path $Dir 'rai.exe'
    $reported = & $rai --version 2>&1
    if ($LASTEXITCODE -ne 0) { Fail "the installed rai.exe does not start on this computer ($reported). Please report it at $RepoUrl/issues" }
    Say "Installed: $Dir"

    # --- PATH, read and written raw so %VARIABLES% in it survive -----------------------
    if ($env:RAI_NO_PATH -ne '1') {
        $key = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment', $true)
        $path = [string]$key.GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
        $entries = @($path.Split(';') | Where-Object { $_ })
        if (-not ($entries | Where-Object { $_.TrimEnd('\') -ieq $Dir.TrimEnd('\') })) {
            $key.SetValue('Path', (($entries + $Dir) -join ';'), [Microsoft.Win32.RegistryValueKind]::ExpandString)
            Say 'Added RAI to your PATH (new terminals will have it).'
        }
        $key.Close()
        # Tell running programs the environment changed, as setx does.
        $signature = '[DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern IntPtr SendMessageTimeout(IntPtr h, uint m, UIntPtr w, string l, uint f, uint t, out UIntPtr r);'
        $user32 = Add-Type -MemberDefinition $signature -Name 'RaiEnvironment' -Namespace 'RaiInstall' -PassThru
        $result = [UIntPtr]::Zero
        [void]$user32::SendMessageTimeout([IntPtr]0xffff, 0x1A, [UIntPtr]::Zero, 'Environment', 2, 5000, [ref]$result)
        if (-not (($env:Path.Split(';')) -contains $Dir)) { $env:Path = "$env:Path;$Dir" }
    }

    # --- Start menu, unless Setup's entry already points here ----------------------------
    if (-not $setupUninstaller) {
        $programs = [Environment]::GetFolderPath('Programs')
        $shell = New-Object -ComObject WScript.Shell
        $link = $shell.CreateShortcut((Join-Path $programs 'RAI Studio.lnk'))
        $link.TargetPath = Join-Path $Dir 'rai-studio.cmd'
        $link.WorkingDirectory = $Dir
        $link.IconLocation = "$rai,0"
        $link.Description = 'Chat with a model on this computer, in your browser'
        $link.Save()
    }
} finally {
    Remove-Item -LiteralPath $Temp -Recurse -Force -ErrorAction SilentlyContinue
}

# --- a model to chat with --------------------------------------------------------------
$home_ = if ($env:RAI_HOME) { $env:RAI_HOME } else { Join-Path $env:USERPROFILE 'RAI' }
$haveModel = Test-Path (Join-Path $home_ 'models\*\*.raimodel')
if ($env:RAI_NO_MODEL -ne '1' -and -not $haveModel) {
    if (-not $Model -and -not [Console]::IsInputRedirected -and [Environment]::UserInteractive) {
        Say ''
        Say 'RAI needs a model to chat with. The recommended one is Qwen2.5 1.5B Instruct:'
        # PowerShell started with -NonInteractive refuses Read-Host; that is
        # a No, not a reason to fail an install that has already finished.
        try { $answer = Read-Host 'a 3.1 GB download. Download it now? [Y/n]' } catch { $answer = 'n' }
        if ($answer -notmatch '^[nN]') { $Model = 'qwen2.5-1.5b' }
    }
    if ($Model) {
        Say ''
        & (Join-Path $Dir 'rai-get-model.cmd') $Model --yes
        if ($LASTEXITCODE -ne 0) { Say "The model download did not finish. Run it again any time with: rai-get-model $Model" }
    }
}

Say ''
Say "RAI $Version is ready."
Say ''
Say '  Chat in your browser:   RAI Studio in the Start menu, or rai-studio'
Say '  Get a model:            rai-get-model'
Say '  Everything else:        rai --help'
Say ''
