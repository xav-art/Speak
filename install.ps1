<#
.SYNOPSIS
    Installs, updates or uninstalls Speak for the current Windows user.

.DESCRIPTION
    No administrator rights are needed. Everything goes into the user profile:

      1. Finds Python 3.10-3.13 (64-bit), or installs Python 3.12 for this user
         with winget, falling back to the signed installer from python.org.
      2. Creates a private virtual environment (.venv) inside the Speak folder
         and installs the packages from requirements.txt (prebuilt wheels only).
      3. Downloads the speech model and the AI cleanup model (one time).
      4. Adds "Speak" to the Desktop and Start Menu, and "speak" to the PATH.

    Two ways to run it:

      * One line in PowerShell (downloads the latest version from GitHub and
        installs it to %LOCALAPPDATA%\Speak):
            irm https://raw.githubusercontent.com/xav-art/Speak/main/install.ps1 | iex

      * From a downloaded or cloned folder: double-click setup.cmd. Speak is
        set up in that folder.

    Running it again updates Speak. Your settings, dictionary and models are kept.

    Optional environment variables (used because "irm | iex" cannot pass
    parameters): SPEAK_INSTALL_DIR, SPEAK_BRANCH, SPEAK_NO_SHORTCUTS=1,
    SPEAK_NO_LAUNCH=1.
#>
param(
    [string] $InstallDir = $env:SPEAK_INSTALL_DIR,
    [string] $Branch = $env:SPEAK_BRANCH,
    [switch] $Uninstall,
    [switch] $NoShortcuts = ($env:SPEAK_NO_SHORTCUTS -eq '1'),
    [switch] $NoLaunch = ($env:SPEAK_NO_LAUNCH -eq '1')
)

# "irm | iex" runs this in the caller's session, so preferences changed here
# are restored at the end instead of leaking into the user's PowerShell.
$SpeakSavedPrefs = @($ErrorActionPreference, $ProgressPreference)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest is very slow with the progress bar on PS 5.1
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$Repo = 'xav-art/Speak'
# Prebuilt CPU wheels of llama-cpp-python (PyPI only has the source, which needs a C++ compiler).
$LlamaWheelIndex = 'https://abetlen.github.io/llama-cpp-python/whl/cpu'
$PythonFallbackUrl = 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe'
$ShortcutName = 'Speak'

function Write-Step([string] $text) { Write-Host ''; Write-Host "==> $text" -ForegroundColor Cyan }
function Write-Ok([string] $text)   { Write-Host "    $text" -ForegroundColor Green }
function Write-Info([string] $text) { Write-Host "    $text" }
function Write-Warn([string] $text) { Write-Host "    $text" -ForegroundColor Yellow }

function Stop-Install([string] $text) {
    Write-Host ''
    Write-Host "Speak could not be installed: $text" -ForegroundColor Red
    Write-Host "If you need help, open an issue at https://github.com/$Repo/issues and paste the text above."
    throw 'Installation stopped.'
}

# ---- Python -----------------------------------------------------------------

function Test-Python([string] $exe) {
    # Accept 64-bit CPython 3.10-3.13 (all dependencies ship prebuilt wheels for these).
    if (-not $exe -or -not (Test-Path -LiteralPath $exe)) { return $false }
    if ($exe -like '*\WindowsApps\*') { return $false }   # Microsoft Store alias, not a real Python
    try {
        $out = & $exe -c "import sys, struct; print(sys.version_info[0], sys.version_info[1], struct.calcsize('P') * 8)" 2>$null
        $major, $minor, $bits = "$out".Trim() -split ' '
        return ($major -eq '3' -and [int]$minor -ge 10 -and [int]$minor -le 13 -and $bits -eq '64')
    } catch { return $false }
}

function Find-Python {
    $candidates = New-Object System.Collections.Generic.List[string]
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        foreach ($v in '3.12', '3.13', '3.11', '3.10') {
            try {
                $exe = & $py.Source "-$v" -c 'import sys; print(sys.executable)' 2>$null
                if ($LASTEXITCODE -eq 0 -and $exe) { $candidates.Add("$exe".Trim()) }
            } catch { }
        }
    }
    foreach ($v in '312', '313', '311', '310') {
        $candidates.Add((Join-Path $env:LOCALAPPDATA "Programs\Python\Python$v\python.exe"))
        $candidates.Add((Join-Path $env:ProgramFiles "Python$v\python.exe"))
    }
    Get-Command python.exe -All -ErrorAction SilentlyContinue | ForEach-Object { $candidates.Add($_.Source) }
    foreach ($c in $candidates) { if (Test-Python $c) { return $c } }
    return $null
}

function Install-Python {
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if ($winget) {
        Write-Info 'Installing Python 3.12 for your user account with winget (no admin rights needed)...'
        try {
            & $winget.Source install --id Python.Python.3.12 --exact --scope user --silent `
                --accept-package-agreements --accept-source-agreements --disable-interactivity | Out-Host
        } catch { Write-Warn "winget failed: $($_.Exception.Message)" }
        $found = Find-Python
        if ($found) { return $found }
        Write-Warn 'winget did not install Python. Trying the official installer from python.org...'
    }

    $setup = Join-Path $env:TEMP 'speak-python-3.12.10-amd64.exe'
    Write-Info "Downloading $PythonFallbackUrl ..."
    Invoke-WebRequest -Uri $PythonFallbackUrl -OutFile $setup -UseBasicParsing
    $sig = Get-AuthenticodeSignature -LiteralPath $setup
    if ($sig.Status -ne 'Valid' -or $sig.SignerCertificate.Subject -notlike '*Python Software Foundation*') {
        Remove-Item -LiteralPath $setup -Force -ErrorAction SilentlyContinue
        Stop-Install 'the downloaded Python installer is not signed by the Python Software Foundation.'
    }
    Write-Info 'Installing Python 3.12 for your user account (no admin rights needed)...'
    $p = Start-Process -FilePath $setup -Wait -PassThru -ArgumentList @(
        '/quiet', 'InstallAllUsers=0', 'PrependPath=0', 'Include_launcher=0',
        'Include_test=0', 'Shortcuts=0', 'AssociateFiles=0')
    Remove-Item -LiteralPath $setup -Force -ErrorAction SilentlyContinue
    if ($p.ExitCode -ne 0) { Stop-Install "the Python installer ended with code $($p.ExitCode)." }
    return Find-Python
}

# ---- Files ------------------------------------------------------------------

function Get-SpeakSource([string] $dest, [string] $branch) {
    # Downloads the repository as a ZIP and copies the app files over $dest.
    # Personal files (config, dictionary) and models are not in the ZIP, so
    # they are never overwritten.
    $zip = Join-Path $env:TEMP "speak-$branch.zip"
    $tmp = Join-Path $env:TEMP "speak-$branch-extract"
    Write-Info "Downloading Speak ($branch) from github.com/$Repo ..."
    Invoke-WebRequest -Uri "https://github.com/$Repo/archive/refs/heads/$branch.zip" -OutFile $zip -UseBasicParsing
    if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Recurse -Force }
    Expand-Archive -LiteralPath $zip -DestinationPath $tmp -Force
    $root = Get-ChildItem -LiteralPath $tmp -Directory | Select-Object -First 1
    if (-not $root -or -not (Test-Path -LiteralPath (Join-Path $root.FullName 'speak.py'))) {
        Stop-Install 'the downloaded archive does not contain speak.py.'
    }
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Copy-Item -Path (Join-Path $root.FullName '*') -Destination $dest -Recurse -Force
    Remove-Item -LiteralPath $zip, $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

function Get-RunningSpeak([string] $dir) {
    $script = Join-Path $dir 'speak.py'
    Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine.IndexOf($script, [StringComparison]::OrdinalIgnoreCase) -ge 0 }
}

function Wait-SpeakClosed([string] $dir) {
    while (Get-RunningSpeak $dir) {
        Write-Warn 'Speak is running. Close the Speak window, then press Enter to continue.'
        [void](Read-Host)
    }
}

# ---- Shortcuts and PATH -------------------------------------------------------

function New-Shortcut([string] $path, [string] $target, [string] $arguments, [string] $workDir, [string] $icon, [string] $description) {
    $ws = New-Object -ComObject WScript.Shell
    $s = $ws.CreateShortcut($path)
    $s.TargetPath = $target
    $s.Arguments = $arguments
    $s.WorkingDirectory = $workDir
    $s.Description = $description
    if ($icon) { $s.IconLocation = $icon }
    $s.Save()
}

function Get-ShortcutPaths {
    $programs = [Environment]::GetFolderPath('Programs')
    [pscustomobject]@{
        Desktop     = Join-Path ([Environment]::GetFolderPath('Desktop')) "$ShortcutName.lnk"
        StartMenu   = Join-Path $programs "$ShortcutName.lnk"
        StartFolder = Join-Path $programs 'Speak'
    }
}

function Set-UserPath([string] $dir, [bool] $add) {
    $current = [Environment]::GetEnvironmentVariable('Path', 'User')
    $parts = @(($current -split ';') | Where-Object { $_ -and ($_.TrimEnd('\') -ine $dir.TrimEnd('\')) })
    if ($add) { $parts += $dir }
    $new = $parts -join ';'
    if ($new -ne $current) { [Environment]::SetEnvironmentVariable('Path', $new, 'User') }
}

# ---- Uninstall --------------------------------------------------------------

function Invoke-Uninstall([string] $dir) {
    Write-Step "Uninstalling Speak from $dir"
    Wait-SpeakClosed $dir
    $sc = Get-ShortcutPaths
    foreach ($p in $sc.Desktop, $sc.StartMenu) {
        if (Test-Path -LiteralPath $p) {
            $target = (New-Object -ComObject WScript.Shell).CreateShortcut($p).TargetPath
            if ($target -and $target.StartsWith($dir, [StringComparison]::OrdinalIgnoreCase)) { Remove-Item -LiteralPath $p -Force }
        }
    }
    if (Test-Path -LiteralPath $sc.StartFolder) { Remove-Item -LiteralPath $sc.StartFolder -Recurse -Force }
    Set-UserPath $dir $false
    Write-Ok 'Shortcuts and PATH entry removed.'

    $answer = Read-Host '    Also delete your personal dictionary and settings? Type YES to delete them, or press Enter to keep them'
    Set-Location $env:TEMP
    $keep = @()
    if ($answer -cne 'YES') { $keep = @('speak_config.json', 'speak_dictionary.json') }
    Get-ChildItem -LiteralPath $dir -Force | Where-Object { $keep -notcontains $_.Name } |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    if (-not (Get-ChildItem -LiteralPath $dir -Force -ErrorAction SilentlyContinue)) {
        Remove-Item -LiteralPath $dir -Force -ErrorAction SilentlyContinue
        Write-Ok 'Speak has been removed.'
    } else {
        Write-Ok "Speak has been removed. Your dictionary and settings are still in $dir"
    }
}

# ---- Install ----------------------------------------------------------------

function Invoke-Install {
    $localSource = $PSScriptRoot -and (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'speak.py'))
    if (-not $InstallDir) {
        $script:InstallDir = if ($localSource) { $PSScriptRoot } else { Join-Path $env:LOCALAPPDATA 'Speak' }
    }
    $dir = [IO.Path]::GetFullPath($InstallDir)
    if (-not $Branch) { $script:Branch = 'main' }

    if ($Uninstall) { Invoke-Uninstall $dir; return }

    Write-Host ''
    Write-Host '  SPEAK  -  private voice dictation for Windows' -ForegroundColor White
    Write-Host "  Installing to: $dir"
    Write-Host '  No admin rights are needed. This takes 5 to 15 minutes the first time.'

    Wait-SpeakClosed $dir

    # The deepest file pip installs (in onnxruntime) adds ~125 characters to
    # this folder. Unless long paths are enabled, Windows stops at 260.
    $longPaths = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -ErrorAction SilentlyContinue).LongPathsEnabled
    if ($dir.Length -gt 125 -and $longPaths -ne 1) {
        Stop-Install "the folder path is too long for Windows ($($dir.Length) characters). Move the Speak folder somewhere shorter, for example $env:USERPROFILE\Speak, and run setup again."
    }

    Write-Step 'Step 1 of 5: Getting Speak'
    if ($localSource -and ([IO.Path]::GetFullPath($PSScriptRoot) -ieq $dir)) {
        Write-Ok 'Using the files in this folder.'
    } elseif ($localSource) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        Copy-Item -Path (Join-Path $PSScriptRoot '*') -Destination $dir -Recurse -Force -Exclude '.git', '.venv', 'models', '__pycache__', 'speak_config.json*', 'speak_dictionary.json*'
        Write-Ok "Copied from $PSScriptRoot"
    } else {
        Get-SpeakSource $dir $Branch
        Write-Ok 'Downloaded.'
    }
    # Files from a downloaded ZIP carry the "Mark of the Web", which makes
    # Windows show a security prompt every time the launcher starts.
    Get-ChildItem -LiteralPath $dir -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notlike '*\.venv\*' -and $_.FullName -notlike '*\models\*' } |
        Unblock-File -ErrorAction SilentlyContinue

    Write-Step 'Step 2 of 5: Checking Python'
    $python = Find-Python
    if ($python) { Write-Ok "Found $python" }
    else {
        $python = Install-Python
        if (-not $python) { Stop-Install 'Python could not be installed. Install Python 3.12 from https://www.python.org/downloads/ (no admin needed) and run this again.' }
        Write-Ok "Installed $python"
    }

    Write-Step 'Step 3 of 5: Installing Speak''s components'
    $venv = Join-Path $dir '.venv'
    $venvPy = Join-Path $venv 'Scripts\python.exe'
    if (-not (Test-Python $venvPy)) {
        if (Test-Path -LiteralPath $venv) { Remove-Item -LiteralPath $venv -Recurse -Force }
        Write-Info 'Creating a private Python environment...'
        & $python -m venv $venv
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $venvPy)) { Stop-Install 'could not create the Python environment.' }
    }
    Write-Info 'Installing packages (about 300 MB)...'
    # Only llama-cpp-python comes from its own wheel index (--no-deps, binary
    # only); everything else, including its dependencies, comes from PyPI.
    & $venvPy -m pip install --disable-pip-version-check --no-input --quiet `
        --no-deps --only-binary :all: --index-url $LlamaWheelIndex 'llama-cpp-python>=0.3.0'
    if ($LASTEXITCODE -eq 0) {
        & $venvPy -m pip install --disable-pip-version-check --no-input --quiet `
            -r (Join-Path $dir 'requirements.txt')
    }
    if ($LASTEXITCODE -ne 0) {
        Stop-Install 'the Python packages could not be installed. On a company network, a proxy or firewall may block pypi.org.'
    }
    Write-Ok 'Components installed.'

    Write-Step 'Step 4 of 5: Downloading the speech and AI models (one time, about 1.3 GB)'
    Push-Location $dir
    try { & $venvPy -X utf8 (Join-Path $dir 'speak.py') --download-models }
    finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) { Write-Warn 'Some models could not be downloaded. Speak will try again when it starts.' }
    else { Write-Ok 'Models ready.' }

    Write-Step 'Step 5 of 5: Adding Speak to your Desktop and Start Menu'
    $launcher = Join-Path $dir 'start_speak.bat'
    $icon = (Join-Path $dir 'assets\speak.ico') + ',0'
    if ($NoShortcuts) {
        Write-Info 'Skipped (SPEAK_NO_SHORTCUTS=1).'
    } else {
        try {
            $sc = Get-ShortcutPaths
            New-Shortcut $sc.Desktop $launcher '' $dir $icon 'Speak - private voice dictation'
            New-Item -ItemType Directory -Force -Path $sc.StartFolder | Out-Null
            New-Shortcut (Join-Path $sc.StartFolder 'Speak.lnk') $launcher '' $dir $icon 'Speak - private voice dictation'
            New-Shortcut (Join-Path $sc.StartFolder 'Uninstall Speak.lnk') (Join-Path $dir 'uninstall.cmd') '' $env:TEMP `
                "$env:SystemRoot\System32\shell32.dll,31" 'Remove Speak from this computer'
            Set-UserPath $dir $true
            Write-Ok 'Desktop and Start Menu shortcuts created. You can also type "speak" in a new terminal.'
        } catch {
            Write-Warn "Shortcuts could not be created ($($_.Exception.Message))."
            Write-Warn "Start Speak by double-clicking $launcher"
        }
    }

    Write-Host ''
    Write-Host '  Speak is installed.' -ForegroundColor Green
    Write-Host '  Start it any time from the Speak icon on your Desktop or in the Start Menu.'
    Write-Host '  Click where you want to write, tap the < key, speak, and tap < again.'
    Write-Host ''
    if (-not $NoLaunch) { Start-Process -FilePath $launcher -WorkingDirectory $dir }
}

try { Invoke-Install }
catch {
    if ($_.Exception.Message -ne 'Installation stopped.') {
        Write-Host ''
        Write-Host "Speak could not be installed: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "If you need help, open an issue at https://github.com/$Repo/issues and paste the text above."
    }
}
finally {
    $ErrorActionPreference, $ProgressPreference = $SpeakSavedPrefs
}
