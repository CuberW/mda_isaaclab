param(
    [Parameter(Position = 0)]
    [ValidateSet(
        "assets",
        "build",
        "shell",
        "health",
        "light",
        "full",
        "headless-all",
        "viewer",
        "wsl-setup",
        "wsl-shell",
        "wsl-health",
        "wsl-full",
        "wsl-headless-all",
        "wsl-viewer"
    )]
    [string]$Command = "shell",

    [Parameter(Position = 1)]
    [ValidateSet("319", "37", "22", "all")]
    [string]$Task = "319"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ComposeFile = Join-Path $ProjectRoot "docker\compose.yaml"
$WslDistro = $env:ROBOT_STACK_WSL_DISTRO
if ([string]::IsNullOrWhiteSpace($WslDistro)) {
    $AvailableDistros = @(wsl -l -q | ForEach-Object {
        ($_ -replace "`0", "").Trim()
    } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($AvailableDistros -contains "UbuntuRobot") {
        $WslDistro = "UbuntuRobot"
    } elseif ($AvailableDistros -contains "Ubuntu") {
        $WslDistro = "Ubuntu"
    } elseif ($AvailableDistros.Count -gt 0) {
        $WslDistro = $AvailableDistros[0]
    } else {
        throw "No WSL distro found. Import UbuntuRobot first or set ROBOT_STACK_WSL_DISTRO."
    }
}
$WslVenv = $env:ROBOT_STACK_VENV
if ([string]::IsNullOrWhiteSpace($WslVenv)) {
    $WslVenv = "/opt/robot-stack/venv"
}
$RosDistro = $env:ROS_DISTRO
if ([string]::IsNullOrWhiteSpace($RosDistro)) {
    $RosDistro = "jazzy"
}

function Invoke-Compose {
    param([string[]]$Args)
    docker compose -f $ComposeFile @Args
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

function Get-WslProjectRoot {
    $windowsPath = $ProjectRoot.Path
    if ($windowsPath -match '^([A-Za-z]):\\(.*)$') {
        $drive = $Matches[1].ToLowerInvariant()
        $rest = $Matches[2] -replace '\\', '/'
        return "/mnt/$drive/$rest"
    }
    $path = (wsl -d $WslDistro -- wslpath -a "$windowsPath").Trim()
    if ([string]::IsNullOrWhiteSpace($path)) {
        throw "Could not resolve project path in WSL: $windowsPath"
    }
    return $path
}

function Quote-Bash {
    param([string]$Text)
    return "'" + $Text.Replace("'", "'`"`"'`"'") + "'"
}

function Invoke-WslProject {
    param([string]$CommandLine)
    $WslRoot = Quote-Bash (Get-WslProjectRoot)
    $TempDir = Join-Path $ProjectRoot.Path ".robot_stack_tmp"
    New-Item -ItemType Directory -Force -Path $TempDir | Out-Null
    $TempName = "wsl-" + [System.Guid]::NewGuid().ToString("N") + ".sh"
    $TempPath = Join-Path $TempDir $TempName
    $Script = @(
        "set -e",
        "cd $WslRoot",
        $CommandLine
    ) -join "`n"
    $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($TempPath, $Script + "`n", $Utf8NoBom)
    $WslScript = (Get-WslProjectRoot).TrimEnd("/") + "/.robot_stack_tmp/$TempName"
    try {
        wsl -d $WslDistro -- bash $WslScript
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    } finally {
        Remove-Item -LiteralPath $TempPath -Force -ErrorAction SilentlyContinue
    }
}

$FullEnvParts = @(
    "source /opt/ros/$RosDistro/setup.bash",
    'if [ -f "$PWD/ros2_ws/install/setup.bash" ]; then source "$PWD/ros2_ws/install/setup.bash"; fi',
    "source $(Quote-Bash $WslVenv)/bin/activate",
    "export CUDA_HOME=/usr/local/cuda",
    "export PATH=/usr/local/cuda/bin:${WslVenv}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/ros/$RosDistro/lib:/opt/ros/$RosDistro/lib/x86_64-linux-gnu",
    "export MUJOCO_GL=egl",
    'export PYTHONPATH="$PWD:$PWD/third_party/graspnet-baseline:$PWD/third_party/graspnet-baseline/models:$PWD/third_party/graspnet-baseline/pointnet2:$PWD/third_party/graspnet-baseline/knn:$PWD/third_party/graspnetAPI:/opt/ros/jazzy/lib/python3.12/site-packages"'
)
$FullEnv = $FullEnvParts -join " && "

switch ($Command) {
    "assets" {
        python (Join-Path $PSScriptRoot "setup_full_stack_assets.py")
    }
    "build" {
        Invoke-Compose @("build")
    }
    "shell" {
        Invoke-Compose @("run", "--rm", "robot-stack", "bash")
    }
    "light" {
        python (Join-Path $PSScriptRoot "verify_system_health.py") --light
    }
    "health" {
        Invoke-Compose @("run", "--rm", "robot-stack", "bash", "-lc", "source /opt/ros/humble/setup.bash && python scripts/verify_system_health.py --full")
    }
    "full" {
        Invoke-Compose @("run", "--rm", "robot-stack", "bash", "-lc", "source /opt/ros/humble/setup.bash && python scripts/verify_system_health.py --full")
    }
    "headless-all" {
        Invoke-Compose @("run", "--rm", "robot-stack", "bash", "-lc", "source /opt/ros/humble/setup.bash && python run_all.py --task 319 --headless && python run_all.py --task 37 --headless --instruction 'pick up the red cup and hand it to me' && python run_all.py --task 22 --headless --instruction 'carry the long rod to the target region'")
    }
    "viewer" {
        if ($Task -eq "all") {
            throw "viewer command needs a single task: 319, 37, or 22"
        }
        Invoke-Compose @("run", "--rm", "robot-stack", "bash", "-lc", "source /opt/ros/humble/setup.bash && python run_all.py --task $Task")
    }
    "wsl-setup" {
        Invoke-WslProject "bash scripts/setup_wsl_full_stack.sh"
    }
    "wsl-shell" {
        Invoke-WslProject "source $(Quote-Bash $WslVenv)/bin/activate && bash"
    }
    "wsl-health" {
        Invoke-WslProject "$FullEnv && python scripts/verify_system_health.py --full"
    }
    "wsl-full" {
        Invoke-WslProject "$FullEnv && python scripts/verify_system_health.py --full"
    }
    "wsl-headless-all" {
        Invoke-WslProject "$FullEnv && python run_all.py --task 319 --headless && python run_all.py --task 37 --headless --instruction 'pick up the red cup and hand it to me' && python run_all.py --task 22 --headless --instruction 'carry the long rod to the target region'"
    }
    "wsl-viewer" {
        if ($Task -eq "all") {
            throw "wsl-viewer command needs a single task: 319, 37, or 22"
        }
        Invoke-WslProject "$FullEnv && python run_all.py --task $Task"
    }
}
