# rai-get-model: download a model from Hugging Face and convert it for RAI.
#
#   rai-get-model                 list the starter models
#   rai-get-model qwen2.5-1.5b    get a starter model by name
#   rai-get-model Qwen/Qwen2.5-1.5B-Instruct
#                                 get any supported model by its Hugging Face id
#
# Options:
#   --keep        keep the downloaded checkpoint after converting it
#   --yes         do not ask before downloading
#   --home DIR    where models live (default: $env:RAI_HOME, or %USERPROFILE%\RAI)
#
# A starter model is pinned to one exact upload, and every file is checked
# against its SHA-256 before it is used. The rai program itself never goes
# online; this script is the only part of RAI that downloads anything, and only
# when you run it. Set HF_TOKEN for models that need a Hugging Face login.
#
# Written for Windows PowerShell 5.1, the version every Windows 10 and 11 has.

$ErrorActionPreference = 'Stop'
# Windows PowerShell 5.1 can default to TLS 1.0, which Hugging Face refuses.
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Endpoint = if ($env:HF_ENDPOINT) { $env:HF_ENDPOINT.TrimEnd('/') } else { 'https://huggingface.co' }
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$RaiHome = if ($env:RAI_HOME) { $env:RAI_HOME } else { Join-Path $env:USERPROFILE 'RAI' }
$Keep = $false
$Yes = $false
$What = ''

function Say([string]$text) { Write-Host $text }
function Fail([string]$text) {
    Write-Host ''
    Write-Host "rai-get-model: $text" -ForegroundColor Red
    Write-Host ''
    exit 1
}

# The same options on every platform, so one set of instructions fits all.
for ($i = 0; $i -lt $args.Count; $i++) {
    $a = [string]$args[$i]
    switch -regex ($a) {
        '^--keep$' { $Keep = $true; continue }
        '^(--yes|-y)$' { $Yes = $true; continue }
        '^--home$' { $i++; if ($i -ge $args.Count) { Fail '--home needs a folder' }; $RaiHome = [string]$args[$i]; continue }
        '^--home=' { $RaiHome = $a.Substring(7); continue }
        '^(-h|--help|/\?)$' {
            Get-Content -LiteralPath $MyInvocation.MyCommand.Path -TotalCount 17 |
                Select-Object -Skip 1 | ForEach-Object { $_ -replace '^# ?', '' }
            exit 0
        }
        '^-' { Fail "unknown option: $a (try --help)" }
        default { if ($What) { Fail 'give one model at a time' }; $What = $a }
    }
}

# --- the starter list -----------------------------------------------------------
$Catalog = $null
foreach ($candidate in @((Join-Path $Here 'models.tsv'), (Join-Path $Here '..\launchers\models.tsv'))) {
    if (Test-Path -LiteralPath $candidate) { $Catalog = $candidate; break }
}
$Rows = @()
if ($Catalog) {
    $Rows = Get-Content -LiteralPath $Catalog | Where-Object { $_ -and -not $_.StartsWith('#') } | ForEach-Object {
        $f = $_ -split "`t"
        [pscustomobject]@{ Name = $f[0]; Repo = $f[1]; Revision = $f[2]; Path = $f[3]; Bytes = [int64]$f[4]
                           Sha256 = $f[5]; ConvertArgs = $f[6]; Description = $f[7] }
    }
}

function Show-Models {
    Say ''
    Say 'Starter models (pinned and checked):'
    Say ''
    if (-not $Rows) { Say '  (the starter list is missing from this install)'; return }
    foreach ($group in ($Rows | Group-Object Name)) {
        $gb = ($group.Group | Measure-Object Bytes -Sum).Sum / 1e9
        Say ('  {0,-14} {1,5:N1} GB download   {2}' -f $group.Name, $gb, $group.Group[0].Description)
    }
    Say ''
    Say 'Get one:    rai-get-model qwen2.5-1.5b'
    Say 'Or any supported Hugging Face model by its id, for example:'
    Say '            rai-get-model HuggingFaceTB/SmolLM2-1.7B-Instruct'
    Say ''
}

if (-not $What) { Show-Models; exit 0 }

# --- tools ------------------------------------------------------------------------
$Curl = Join-Path ([Environment]::SystemDirectory) 'curl.exe'
if (-not (Test-Path -LiteralPath $Curl)) { $Curl = $null }

$Headers = @{}
if ($env:HF_TOKEN) { $Headers['Authorization'] = "Bearer $($env:HF_TOKEN)" }

# Download $Path from the repository to $Target, resuming a partial file.
function Get-RepoFile([string]$Path, [string]$Target) {
    $url = "$Endpoint/$Repo/resolve/$Revision/$Path"
    $part = "$Target.part"
    if ($Curl) {
        $curlArgs = @('-L', '-#', '-C', '-', '--retry', '5', '--retry-delay', '5', '-w', '%{http_code}', '-o', $part)
        if ($env:HF_TOKEN) { $curlArgs += @('-H', "Authorization: Bearer $($env:HF_TOKEN)") }
        $code = & $Curl @curlArgs $url
        if ($LASTEXITCODE -ne 0) { Fail "downloading $Path was interrupted. Run the same command again to resume." }
    } else {
        # Windows before 1803 has no curl.exe; this way cannot resume.
        try {
            Invoke-WebRequest -Uri $url -OutFile $part -Headers $Headers -UseBasicParsing
            $code = '200'
        } catch {
            $code = [string][int]$_.Exception.Response.StatusCode
        }
    }
    switch -regex ($code) {
        '^(200|206|416)$' { Move-Item -LiteralPath $part -Destination $Target -Force }
        '^(401|403)$' {
            Remove-Item -LiteralPath $part -ErrorAction SilentlyContinue
            Fail ("Hugging Face refused $Repo ($code). This model needs a login:`n" +
                  "    1. Open https://huggingface.co/$Repo and accept its license.`n" +
                  "    2. Create a token at https://huggingface.co/settings/tokens`n" +
                  "    3. Run again after: `$env:HF_TOKEN = '<your token>'")
        }
        '^404$' { Remove-Item -LiteralPath $part -ErrorAction SilentlyContinue; Fail "$Repo has no file $Path at $Revision (404)." }
        default { Remove-Item -LiteralPath $part -ErrorAction SilentlyContinue; Fail "downloading $Path failed (HTTP $code). Run the same command again." }
    }
}

# Through .NET rather than Get-FileHash, which Windows PowerShell started from
# PowerShell 7 loses with the module path it inherits.
function Get-Sha256([string]$Path) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $stream = [System.IO.File]::OpenRead($Path)
    try { ([System.BitConverter]::ToString($sha.ComputeHash($stream)) -replace '-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $sha.Dispose() }
}

# --- what to download -------------------------------------------------------------
$Files = @()
$ConvertArgs = @()
$Pinned = $false
if ($What.Contains('/')) {
    $Repo = $What
    $Name = ($What.Split('/')[-1]).ToLowerInvariant()
    $Revision = 'main'
} else {
    $mine = @($Rows | Where-Object { $_.Name -eq $What })
    if (-not $mine) { Show-Models; Fail "no starter model is called '$What'" }
    $Repo = $mine[0].Repo
    $Name = $What
    $Revision = $mine[0].Revision
    if ($mine[0].ConvertArgs) { $ConvertArgs = $mine[0].ConvertArgs -split ' ' }
    $Files = $mine | ForEach-Object { [pscustomobject]@{ Path = $_.Path; Bytes = $_.Bytes; Sha256 = $_.Sha256 } }
    $Pinned = $true
}

$Rai = $null
foreach ($candidate in @((Join-Path $Here 'rai.exe'), (Join-Path $Here '..\rai.exe'), (Join-Path $Here '..\bin\rai.exe'))) {
    if (Test-Path -LiteralPath $candidate) { $Rai = (Resolve-Path -LiteralPath $candidate).Path; break }
}
if (-not $Rai) {
    $found = Get-Command rai.exe -ErrorAction SilentlyContinue
    if ($found) { $Rai = $found.Source } else { Fail 'the rai program was not found. Put this script next to rai.exe, or put rai on your PATH.' }
}

$ModelDir = Join-Path (Join-Path $RaiHome 'models') $Name
$DownloadDir = Join-Path (Join-Path $RaiHome 'downloads') $Name
$Output = Join-Path $ModelDir "$Name-q4.raimodel"
if (Test-Path -LiteralPath $Output) {
    Say ''
    Say "Already converted: $Output"
    Say ''
    Say 'Chat with it in your browser:   rai-studio'
    Say "Or in the terminal:             rai run `"$Output`" --prompt `"Hello`""
    Say ''
    exit 0
}
New-Item -ItemType Directory -Force -Path $DownloadDir, $ModelDir | Out-Null

if (-not $Pinned) {
    # Ask Hugging Face for the file list, take what the converter reads, and
    # check the configuration before downloading gigabytes of weights.
    Say "Looking up $Repo on Hugging Face..."
    try {
        $info = Invoke-RestMethod -Uri "$Endpoint/api/models/$Repo" -Headers $Headers -UseBasicParsing
    } catch {
        Fail "Hugging Face has no model $Repo, or it needs a login (set `$env:HF_TOKEN)."
    }
    if ($info.sha) { $Revision = $info.sha }
    $wanted = '^(config\.json|generation_config\.json|tokenizer\.json|tokenizer_config\.json|special_tokens_map\.json|chat_template\.jinja|chat_template\.json|model\.safetensors\.index\.json|LICENSE|[^/]+\.safetensors)$'
    $Files = @($info.siblings | Where-Object { $_.rfilename -match $wanted } |
        ForEach-Object { [pscustomobject]@{ Path = $_.rfilename; Bytes = 0; Sha256 = '' } })
    if (-not ($Files | Where-Object { $_.Path -eq 'config.json' })) { Fail "$Repo has no config.json, so it is not a model RAI can convert." }
    if (-not ($Files | Where-Object { $_.Path -like '*.safetensors' })) { Fail "$Repo has no .safetensors weights. RAI converts safetensors checkpoints only." }

    Say 'Checking that RAI can convert it...'
    foreach ($small in 'config.json', 'tokenizer_config.json', 'generation_config.json') {
        if ($Files | Where-Object { $_.Path -eq $small }) { Get-RepoFile $small (Join-Path $DownloadDir $small) }
    }
    & $Rai check $DownloadDir
    if ($LASTEXITCODE -ne 0) { Fail "RAI cannot run $Repo (the reason is above). Nothing large was downloaded. See the starter models with: rai-get-model" }
}

# --- disk space ---------------------------------------------------------------------
$need = [int64](($Files | Measure-Object Bytes -Sum).Sum * 1.4)
$drive = $null
try {
    $drive = New-Object System.IO.DriveInfo ([System.IO.Path]::GetPathRoot((Resolve-Path -LiteralPath $RaiHome).Path))
} catch {
    # A network share has no drive to ask; the download then reports any shortfall itself.
}
if ($need -gt 0 -and $drive) {
    if ($drive.AvailableFreeSpace -lt $need) {
        Fail ("not enough free space on {0}: the download and the converted model need about {1:N0} GB, and {2:N0} GB is free." -f
              $drive.Name, ($need / 1GB + 1), ($drive.AvailableFreeSpace / 1GB))
    }
}

Say ''
Say "Model:     $Repo"
if ($Pinned) {
    Say ('Download:  {0:N1} GB, pinned to {1} and checked file by file' -f (($Files | Measure-Object Bytes -Sum).Sum / 1e9), $Revision.Substring(0, 12))
}
Say "Saved to:  $ModelDir"
if (-not $Yes -and -not [Console]::IsInputRedirected) {
    $answer = Read-Host 'Download it now? [Y/n]'
    if ($answer -match '^[nN]') { Say 'Nothing downloaded.'; exit 1 }
}

# --- download and check every file -------------------------------------------------
foreach ($file in $Files) {
    $target = Join-Path $DownloadDir $file.Path
    if ((Test-Path -LiteralPath $target) -and $file.Sha256 -and ((Get-Sha256 $target) -eq $file.Sha256)) {
        Say "  $($file.Path): already downloaded"
        continue
    }
    Say "  $($file.Path)"
    Get-RepoFile $file.Path $target
    if ($file.Sha256) {
        $actual = Get-Sha256 $target
        if ($actual -ne $file.Sha256) {
            Remove-Item -LiteralPath $target
            Fail ("$($file.Path) is not the file that was tested (SHA-256 $actual, expected $($file.Sha256)).`n" +
                  '    It was deleted. Run the same command again to download it fresh.')
        }
    }
}

# --- convert ---------------------------------------------------------------------------
Say ''
Say "Converting to RAI's format (this takes a minute or two)..."
& $Rai convert $DownloadDir -o $Output @ConvertArgs
if ($LASTEXITCODE -ne 0) {
    Remove-Item -LiteralPath $Output -ErrorAction SilentlyContinue
    Fail "the conversion failed (the reason is above). The download is kept in $DownloadDir."
}
$license = Join-Path $DownloadDir 'LICENSE'
if (Test-Path -LiteralPath $license) { Copy-Item -LiteralPath $license -Destination (Join-Path $ModelDir 'LICENSE') -Force }

if (-not $Keep) {
    Remove-Item -LiteralPath $DownloadDir -Recurse -Force
    $downloads = Join-Path $RaiHome 'downloads'
    if (-not (Get-ChildItem -LiteralPath $downloads -Force -ErrorAction SilentlyContinue)) {
        Remove-Item -LiteralPath $downloads -ErrorAction SilentlyContinue
    }
}

Say ''
Say "Ready: $Output"
Say ''
Say 'Chat with it in your browser:   rai-studio'
Say "Or in the terminal:             rai run `"$Output`" --prompt `"Hello`""
Say ''
exit 0
