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
# Written for Windows PowerShell 5.1, the version every Windows 10 and 11 has,
# and runs in PowerShell 7 as well.

# One script block, so `irm | iex` leaves nothing behind in the window it runs
# in: no changed preferences, variables or functions.
& {
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
# .NET before 4.7 offers TLS 1.0 alone, which GitHub refuses: TLS 1.2 is added
# there. SystemDefault (0) already lets Windows choose it and is left alone.
$tls = [Net.ServicePointManager]::SecurityProtocol
if ([int]$tls -ne 0 -and -not ($tls -band [Net.SecurityProtocolType]::Tls12)) {
    [Net.ServicePointManager]::SecurityProtocol = $tls -bor [Net.SecurityProtocolType]::Tls12
}

$RepoUrl = 'https://github.com/Classevelabs/rai'
$Target = 'x86_64-pc-windows-msvc'
$Version = if ($env:RAI_VERSION) { $env:RAI_VERSION.TrimStart('v') } else { '' }
$Dir = if ($env:RAI_INSTALL_DIR) { $env:RAI_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA 'Programs\RAI' }
# A full path with no trailing backslash, however it was typed: relative to
# this window's folder, with a slash at the end, or with forward slashes.
$Dir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Dir)
$Dir = $Dir.Replace('/', '\')
if ($Dir.Length -gt 3) { $Dir = $Dir.TrimEnd('\') }
# A folder that already holds the release files, used instead of the internet.
# This is how the release is tested on clean machines before it is published.
$From = $env:RAI_INSTALL_FROM
$Model = $env:RAI_INSTALL_MODEL
# RAI's install folder carries this file. Only an empty folder or one that has
# it is ever replaced, so RAI_INSTALL_DIR pointed at a folder of other things
# can never empty it. Setup writes it too.
$Mark = '.rai-install'

function Say([string]$text) { Write-Host $text }
# Through .NET rather than Get-FileHash and Expand-Archive: Windows PowerShell
# started from PowerShell 7 inherits PowerShell 7's module path and loses both.
function Get-Sha256([string]$path) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $stream = [System.IO.File]::OpenRead($path)
    try { ([System.BitConverter]::ToString($sha.ComputeHash($stream)) -replace '-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $sha.Dispose() }
}
function Fail([string]$text) {
    Write-Host ''
    Write-Host "RAI installer: $text" -ForegroundColor Red
    Write-Host ''
    # `exit` inside `irm | iex` would close the user's PowerShell window.
    throw 'RAI was not installed.'
}

if (-not [Environment]::Is64BitOperatingSystem) { Fail 'RAI needs 64-bit Windows.' }
$Parent = Split-Path -Parent $Dir
$Leaf = Split-Path -Leaf $Dir
if (-not $Parent -or -not $Leaf) { Fail "$Dir is a drive, not a folder. Choose a folder with RAI_INSTALL_DIR." }

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
if ($Version -notmatch '^\d+\.\d+\.\d+[0-9A-Za-z.-]*$') { Fail 'could not find out the latest RAI version. Set $env:RAI_VERSION, for example to 0.2.6, and run this again.' }

$Archive = "rai-$Version-$Target.zip"
$Temp = Join-Path ([System.IO.Path]::GetTempPath()) ("rai-install-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $Temp | Out-Null
$setupInstall = $false

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
    $actual = Get-Sha256 (Join-Path $Temp $Archive)
    if ($actual -ne $expected) { Fail "the download does not match SHA256SUMS (got $actual, expected $expected). Nothing was installed." }
    Say 'Checked: SHA-256 matches the release.'

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory((Join-Path $Temp $Archive), $Temp)
    $unpacked = Join-Path $Temp "rai-$Version-$Target"
    if (-not (Test-Path -LiteralPath (Join-Path $unpacked 'rai.exe'))) { Fail 'the archive has no rai.exe in it.' }

    if ((Test-Path -LiteralPath $Dir) -and -not (Test-Path -LiteralPath (Join-Path $Dir $Mark)) -and
        (Get-ChildItem -LiteralPath $Dir -Force | Select-Object -First 1)) {
        Fail "$Dir already holds other files and is not an earlier RAI install. Choose an empty folder with RAI_INSTALL_DIR."
    }
    # A running rai.exe (RAI Studio left open) holds its folder.
    $running = Get-Process -Name rai -ErrorAction SilentlyContinue | Where-Object {
        try { $_.Path -and $_.Path.StartsWith($Dir + '\', [StringComparison]::OrdinalIgnoreCase) } catch { $false }
    }
    if ($running) { Fail 'RAI is running (RAI Studio may be open). Close it, then run this again.' }

    # The program and its launchers in one folder, so one PATH entry covers
    # rai, rai-studio and rai-get-model. It is assembled beside the install
    # and swapped in by renaming: a folder still in use stops the install
    # before anything is removed, and an interruption leaves the earlier
    # install whole.
    New-Item -ItemType Directory -Force -Path $Parent | Out-Null
    $stage = Join-Path $Parent ".$Leaf.new"
    $old = Join-Path $Parent ".$Leaf.old"
    foreach ($leftover in $stage, $old) {
        if (Test-Path -LiteralPath $leftover) { Remove-Item -LiteralPath $leftover -Recurse -Force }
    }
    New-Item -ItemType Directory -Path $stage | Out-Null
    Get-ChildItem -LiteralPath $unpacked -File | Copy-Item -Destination $stage
    foreach ($name in 'rai-studio.cmd', 'rai-get-model.cmd', 'rai-get-model-windows.ps1', 'models.tsv') {
        Copy-Item -LiteralPath (Join-Path (Join-Path $unpacked 'launchers') $name) -Destination $stage
    }
    # A folder RAI's Setup made keeps Setup's uninstaller, which Settings >
    # Apps runs, and Setup's Start menu entry: one entry, and an uninstaller
    # that still removes everything.
    $setupInstall = Test-Path -LiteralPath (Join-Path $Dir 'uninstall.exe')
    if ($setupInstall) { Copy-Item -LiteralPath (Join-Path $Dir 'uninstall.exe') -Destination $stage }
    [System.IO.File]::WriteAllText((Join-Path $stage $Mark), "$Version`r`n")

    $reported = & (Join-Path $stage 'rai.exe') --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
        Fail "the new rai.exe does not start on this computer ($reported). Nothing was changed. Please report it at $RepoUrl/issues"
    }

    if (Test-Path -LiteralPath $Dir) {
        try { Rename-Item -LiteralPath $Dir -NewName ".$Leaf.old" }
        catch {
            Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
            Fail "$Dir is in use. Close RAI Studio and any window or terminal open in that folder, then run this again. Nothing was changed."
        }
    }
    try { Rename-Item -LiteralPath $stage -NewName $Leaf }
    catch {
        if (Test-Path -LiteralPath $old) { Rename-Item -LiteralPath $old -NewName $Leaf -ErrorAction SilentlyContinue }
        Fail "could not put the new install in place ($($_.Exception.Message)). The earlier one is back."
    }
    if (Test-Path -LiteralPath $old) { Remove-Item -LiteralPath $old -Recurse -Force -ErrorAction SilentlyContinue }
    $rai = Join-Path $Dir 'rai.exe'
    Say "Installed: $Dir"

    # --- PATH, read and written raw so %VARIABLES% in it survive -----------------------
    if ($env:RAI_NO_PATH -ne '1') {
        $key = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment', $true)
        $path = [string]$key.GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
        $entries = @($path.Split(';') | Where-Object { $_ })
        if (-not ($entries | Where-Object { $_.TrimEnd('\') -ieq $Dir })) {
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
    if (-not $setupInstall) {
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
}
