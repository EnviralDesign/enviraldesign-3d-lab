Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Pick-InputImage {
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title = "Choose an image for TRELLIS.2"
    $dialog.Filter = "Images|*.png;*.jpg;*.jpeg;*.webp;*.bmp|All files|*.*"
    if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { return $dialog.FileName }
    return $null
}

function Pick-ModelFolder {
    $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
    $dialog.Description = "Choose a local model folder with pipeline.json"
    if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { return $dialog.SelectedPath }
    return $null
}

function Pick-File {
    param([string]$Title, [string]$Filter)
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title = $Title
    $dialog.Filter = $Filter
    if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { return $dialog.FileName }
    return $null
}

function Start-Viewer {
    param([string]$GlbPath)
    try {
        $port = 8765
        $serverReady = $false
        $client = New-Object Net.Sockets.TcpClient
        $client.Connect("127.0.0.1", $port)
        $client.Close()
        $serverReady = $true
    } catch {
        $serverReady = $false
    }

    try {
        if (-not $serverReady) {
            Start-Process -WindowStyle Hidden -FilePath ".venv\Scripts\python.exe" -ArgumentList @("-m", "http.server", "$port", "--bind", "127.0.0.1")
            Start-Sleep -Seconds 2
        }

        $rootPath = (Resolve-Path $Root).Path
        $glbFullPath = (Resolve-Path $GlbPath).Path
        $rootUri = New-Object System.Uri(($rootPath.TrimEnd('\') + '\'))
        $glbUri = New-Object System.Uri($glbFullPath)
        $relative = [System.Uri]::UnescapeDataString($rootUri.MakeRelativeUri($glbUri).ToString())
        $src = [System.Uri]::EscapeDataString("/$relative")
        Start-Process "http://127.0.0.1:$port/tools/glb_viewer.html?src=$src"
    } catch {
        Write-Host "Viewer launch failed, but the GLB was generated: $GlbPath"
        Write-Host $_.Exception.Message
    }
}

function Open-StageArtifactsFromText {
    param([string]$Text)
    if (-not $script:openStageViewers -or -not $script:openStageViewers.Checked) { return }
    foreach ($line in ($Text -split "`r?`n")) {
        if ($line -match '^STAGE_ARTIFACT\t([^\t]+)\t(.+)$') {
            $label = $matches[1]
            $path = $matches[2].Trim()
            if (-not $script:openedArtifacts) { $script:openedArtifacts = @{} }
            if (-not $script:openedArtifacts.ContainsKey($path) -and (Test-Path $path)) {
                $script:openedArtifacts[$path] = $true
                Append-Log "Opening stage viewer: $label"
                Start-Viewer -GlbPath $path
            }
        }
    }
}

function Quote-CommandArgument {
    param([string]$Value)
    if ($null -eq $Value) { return '""' }
    $escaped = $Value.Replace('"', '\"')
    if ($escaped.EndsWith('\')) { $escaped = $escaped + '\' }
    return [string]::Concat('"', $escaped, '"')
}

function Append-Log {
    param([string]$Text)
    Append-LogText ($Text + [Environment]::NewLine)
}

function Append-LogText {
    param([string]$Text)
    if (-not $script:logBox) { return }
    $action = {
        param($text)
        if ($script:logBox.IsDisposed) { return }
        $script:logBox.AppendText($text)
        $script:logBox.SelectionStart = $script:logBox.TextLength
        $script:logBox.ScrollToCaret()
    }
    if ($script:logBox.InvokeRequired) {
        [void]$script:logBox.BeginInvoke($action, @($Text))
    } else {
        & $action $Text
    }
}

function Update-RunLog {
    if (-not $script:currentLog -or -not (Test-Path $script:currentLog)) { return }
    $fs = $null
    try {
        $fs = [System.IO.File]::Open($script:currentLog, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
        if ($fs.Length -le $script:logReadOffset) { return }
        [void]$fs.Seek($script:logReadOffset, [System.IO.SeekOrigin]::Begin)
        $count = [int]($fs.Length - $script:logReadOffset)
        $bytes = New-Object byte[] $count
        $read = $fs.Read($bytes, 0, $count)
        if ($read -gt 0) {
            $script:logReadOffset += $read
            $text = [System.Text.Encoding]::Default.GetString($bytes, 0, $read)
            Append-LogText $text
            Open-StageArtifactsFromText $text
        }
    } catch {
        # The child process may still be creating or flushing the log file.
    } finally {
        if ($fs) { $fs.Close() }
    }
}

function Set-RunningState {
    param([bool]$Running)
    $run.Enabled = -not $Running
    $cancel.Text = if ($Running) { "Stop" } else { "Close" }
    $statusLabel.Text = if ($Running) { "Running TRELLIS.2 generation..." } else { "Idle" }
}

function Add-Label {
    param($Parent, [string]$Text, [int]$X, [int]$Y, [string]$Tip)
    $label = New-Object System.Windows.Forms.Label
    $label.Text = $Text
    $label.Location = New-Object System.Drawing.Point($X, $Y)
    $label.Size = New-Object System.Drawing.Size(172, 20)
    $Parent.Controls.Add($label)
    if ($Tip) { $script:ToolTip.SetToolTip($label, $Tip) }
    return $label
}

function Add-Number {
    param($Parent, [string]$Name, [int]$X, [int]$Y, [decimal]$Value, [decimal]$Min, [decimal]$Max, [int]$Decimals, [decimal]$Increment, [string]$Tip)
    Add-Label $Parent $Name $X $Y $Tip | Out-Null
    $box = New-Object System.Windows.Forms.NumericUpDown
    $box.Location = New-Object System.Drawing.Point(($X + 180), ($Y - 2))
    $box.Size = New-Object System.Drawing.Size(95, 24)
    $box.Minimum = $Min
    $box.Maximum = $Max
    $box.DecimalPlaces = $Decimals
    $box.Increment = $Increment
    $box.Value = $Value
    $Parent.Controls.Add($box)
    if ($Tip) { $script:ToolTip.SetToolTip($box, $Tip) }
    return $box
}

function Add-Text {
    param($Parent, [string]$Name, [int]$X, [int]$Y, [string]$Value, [int]$Width, [string]$Tip)
    Add-Label $Parent $Name $X $Y $Tip | Out-Null
    $box = New-Object System.Windows.Forms.TextBox
    $box.Location = New-Object System.Drawing.Point(($X + 180), ($Y - 2))
    $box.Size = New-Object System.Drawing.Size($Width, 24)
    $box.Text = $Value
    $Parent.Controls.Add($box)
    if ($Tip) { $script:ToolTip.SetToolTip($box, $Tip) }
    return $box
}

function Add-Combo {
    param($Parent, [string]$Name, [int]$X, [int]$Y, [string[]]$Items, [string]$Value, [int]$Width, [string]$Tip)
    Add-Label $Parent $Name $X $Y $Tip | Out-Null
    $combo = New-Object System.Windows.Forms.ComboBox
    $combo.Location = New-Object System.Drawing.Point(($X + 180), ($Y - 2))
    $combo.Size = New-Object System.Drawing.Size($Width, 24)
    $combo.DropDownStyle = "DropDownList"
    foreach ($item in $Items) { [void]$combo.Items.Add($item) }
    $combo.SelectedItem = $Value
    $Parent.Controls.Add($combo)
    if ($Tip) { $script:ToolTip.SetToolTip($combo, $Tip) }
    return $combo
}

function Add-StageControls {
    param($Parent, [string]$Title, [int]$X, [int]$Y, [hashtable]$Defaults)
    $group = New-Object System.Windows.Forms.GroupBox
    $group.Text = $Title
    $group.Location = New-Object System.Drawing.Point($X, $Y)
    $group.Size = New-Object System.Drawing.Size(300, 245)
    $Parent.Controls.Add($group)

    $controls = @{}
    $controls.group = $group
    $controls.steps = Add-Number $group "Sampling steps" 12 28 $Defaults.steps 1 50 0 1 "More denoising/sampling steps usually improve quality and stability, but increase runtime and VRAM pressure."
    $controls.guidance = Add-Number $group "Guidance strength" 12 58 $Defaults.guidance 1 10 1 0.1 "How strongly the model follows the input image. Too high can add artifacts; too low can drift."
    $controls.rescale = Add-Number $group "Guidance rescale" 12 88 $Defaults.rescale 0 1 2 0.01 "Dampens over-strong classifier-free guidance. Upstream defaults: sparse 0.7, shape 0.5, texture 0.0."
    $controls.rescaleT = Add-Number $group "Rescale T" 12 118 $Defaults.rescaleT 1 6 1 0.1 "Sampler time rescale parameter from the upstream app. Higher changes the denoising schedule strength."
    $controls.intervalStart = Add-Number $group "Guidance start" 12 148 $Defaults.intervalStart 0 1 2 0.01 "Fraction of the sampling schedule where classifier-free guidance begins. Comfy advanced workflows often use 0.10 for sparse/shape."
    $controls.intervalEnd = Add-Number $group "Guidance end" 12 178 $Defaults.intervalEnd 0 1 2 0.01 "Fraction of the sampling schedule where classifier-free guidance ends. Texture workflows often stop around 0.90."
    return $controls
}

$script:ToolTip = New-Object System.Windows.Forms.ToolTip
$script:ToolTip.AutoPopDelay = 20000
$script:ToolTip.InitialDelay = 350
$script:ToolTip.ReshowDelay = 100
$script:ToolTip.ShowAlways = $true

$presets = @(
    [pscustomobject]@{ Name="RTX 5000 standard 512"; Gpu="0"; Pipeline="512"; Steps=12; MaxTokens=24576; Decimation=500000; Texture=2048; SparseRes=0; SaveStage1=$true; SaveStage2=$true; SaveStage3=$true; OpenStageViewers=$true; StopAfter="none"; NoExport=$false; FillHoles=$true; ExportRemesh=$true; RemeshBand=1.0; RemeshProject=0.9; PreSimplify=16777216; Note="Baseline happy-medium run for this machine." },
    [pscustomobject]@{ Name="RTX 5000 inspect stage 1"; Gpu="0"; Pipeline="512"; Steps=12; MaxTokens=24576; Decimation=500000; Texture=2048; SparseRes=0; SaveStage1=$true; SaveStage2=$false; SaveStage3=$false; OpenStageViewers=$true; StopAfter="stage1"; NoExport=$false; FillHoles=$true; ExportRemesh=$true; RemeshBand=1.0; RemeshProject=0.9; PreSimplify=16777216; Note="Fast silhouette/background check only." },
    [pscustomobject]@{ Name="RTX 5000 inspect stage 2"; Gpu="0"; Pipeline="512"; Steps=12; MaxTokens=24576; Decimation=500000; Texture=2048; SparseRes=0; SaveStage1=$true; SaveStage2=$true; SaveStage3=$false; OpenStageViewers=$true; StopAfter="stage2"; NoExport=$false; FillHoles=$true; ExportRemesh=$true; RemeshBand=1.0; RemeshProject=0.9; PreSimplify=16777216; Note="Stops after raw shape geometry." },
    [pscustomobject]@{ Name="RTX 5000 inspect stage 3"; Gpu="0"; Pipeline="512"; Steps=12; MaxTokens=24576; Decimation=500000; Texture=2048; SparseRes=0; SaveStage1=$true; SaveStage2=$true; SaveStage3=$true; OpenStageViewers=$true; StopAfter="stage3"; NoExport=$false; FillHoles=$true; ExportRemesh=$true; RemeshBand=1.0; RemeshProject=0.9; PreSimplify=16777216; Note="Stops before final o-voxel export." },
    [pscustomobject]@{ Name="RTX 5000 export no remesh"; Gpu="0"; Pipeline="512"; Steps=12; MaxTokens=24576; Decimation=1000000; Texture=2048; SparseRes=0; SaveStage1=$true; SaveStage2=$true; SaveStage3=$true; OpenStageViewers=$true; StopAfter="none"; NoExport=$false; FillHoles=$true; ExportRemesh=$false; RemeshBand=1.0; RemeshProject=0.9; PreSimplify=0; Note="Full run testing the non-remesh GLB export path." },
    [pscustomobject]@{ Name="RTX 5000 high mesh export"; Gpu="0"; Pipeline="512"; Steps=12; MaxTokens=24576; Decimation=1000000; Texture=2048; SparseRes=0; SaveStage1=$true; SaveStage2=$true; SaveStage3=$true; OpenStageViewers=$true; StopAfter="none"; NoExport=$false; FillHoles=$true; ExportRemesh=$true; RemeshBand=1.0; RemeshProject=0.9; PreSimplify=0; Note="Baseline generation with heavier final mesh preservation." },
    [pscustomobject]@{ Name="RTX 5000 FaithC after stage 2"; Gpu="0"; Pipeline="512"; Steps=12; MaxTokens=24576; Decimation=500000; Texture=2048; SparseRes=0; SaveStage1=$true; SaveStage2=$true; SaveStage3=$false; OpenStageViewers=$true; StopAfter="stage2"; NoExport=$false; FillHoles=$true; ExportRemesh=$true; RemeshBand=1.0; RemeshProject=0.9; PreSimplify=16777216; FaithCMode="after-stage2"; FaithCResolution=256; Note="Stops after Stage 2 and runs FaithC on the raw shape mesh." },
    [pscustomobject]@{ Name="RTX 5000 FaithC after final"; Gpu="0"; Pipeline="512"; Steps=12; MaxTokens=24576; Decimation=1000000; Texture=2048; SparseRes=0; SaveStage1=$true; SaveStage2=$true; SaveStage3=$true; OpenStageViewers=$true; StopAfter="none"; NoExport=$false; FillHoles=$true; ExportRemesh=$true; RemeshBand=1.0; RemeshProject=0.9; PreSimplify=0; FaithCMode="after-final"; FaithCResolution=256; Note="Runs FaithC as a final contour/export comparison." },
    [pscustomobject]@{ Name="3070 smoke test"; Gpu="1"; Pipeline="512"; Steps=1; MaxTokens=8192; Decimation=100000; Texture=1024; SparseRes=0; SaveStage1=$true; SaveStage2=$false; SaveStage3=$false; OpenStageViewers=$true; StopAfter="none"; NoExport=$false; FillHoles=$true; ExportRemesh=$true; RemeshBand=1.0; RemeshProject=0.9; PreSimplify=16777216; Note="Fast proof-of-life. Expect rough geometry." },
    [pscustomobject]@{ Name="3070 better 512"; Gpu="1"; Pipeline="512"; Steps=12; MaxTokens=12288; Decimation=300000; Texture=1024; SparseRes=0; SaveStage1=$true; SaveStage2=$true; SaveStage3=$true; OpenStageViewers=$true; StopAfter="none"; NoExport=$false; FillHoles=$true; ExportRemesh=$true; RemeshBand=1.0; RemeshProject=0.9; PreSimplify=16777216; Note="More faithful 512 test for the 8 GB card." },
    [pscustomobject]@{ Name="RTX 5000 1024 cascade"; Gpu="0"; Pipeline="1024_cascade"; Steps=12; MaxTokens=49152; Decimation=500000; Texture=2048; SparseRes=0; SaveStage1=$true; SaveStage2=$true; SaveStage3=$true; OpenStageViewers=$true; StopAfter="none"; NoExport=$false; FillHoles=$true; ExportRemesh=$true; RemeshBand=1.0; RemeshProject=0.9; PreSimplify=16777216; Note="Closest to the official app default quality path." },
    [pscustomobject]@{ Name="PixelArtistry UltraShape03-ish"; Gpu="0"; Pipeline="1024_cascade"; SparseSteps=30; SparseGuidance=6.6; SparseRescale=0.1; SparseRescaleT=1.0; ShapeSteps=30; ShapeGuidance=8.2; ShapeRescale=0.1; ShapeRescaleT=2.0; TexSteps=12; TexGuidance=1.0; TexRescale=0.0; TexRescaleT=3.0; SparseStart=0.1; SparseEnd=1.0; ShapeStart=0.1; ShapeEnd=1.0; TexStart=0.0; TexEnd=0.9; MaxTokens=4550; SparseRes=32; Decimation=1000000; Texture=4096; SaveStage1=$true; SaveStage2=$true; SaveStage3=$true; OpenStageViewers=$true; StopAfter="none"; NoExport=$false; FillHoles=$true; ExportRemesh=$true; RemeshBand=1.0; RemeshProject=0.9; PreSimplify=0; Note="Comfy sampler values that exist in this local runner; UltraShape itself is not included." }
)

$form = New-Object System.Windows.Forms.Form
$form.Text = "TRELLIS.2 Launcher"
$form.Size = New-Object System.Drawing.Size(1010, 780)
$form.MinimumSize = New-Object System.Drawing.Size(900, 700)
$form.StartPosition = "CenterScreen"

$tabs = New-Object System.Windows.Forms.TabControl
$tabs.Location = New-Object System.Drawing.Point(12, 12)
$tabs.Size = New-Object System.Drawing.Size(970, 650)
$tabs.Anchor = "Top,Bottom,Left,Right"
$form.Controls.Add($tabs)

$generalTab = New-Object System.Windows.Forms.TabPage
$generalTab.Text = "Setup"
$generalTab.AutoScroll = $true
$tabs.TabPages.Add($generalTab)

$stagesTab = New-Object System.Windows.Forms.TabPage
$stagesTab.Text = "Advanced Samplers"
$stagesTab.AutoScroll = $true
$tabs.TabPages.Add($stagesTab)

$modulesTab = New-Object System.Windows.Forms.TabPage
$modulesTab.Text = "Post Modules"
$modulesTab.AutoScroll = $true
$tabs.TabPages.Add($modulesTab)

$logTab = New-Object System.Windows.Forms.TabPage
$logTab.Text = "Run Log"
$tabs.TabPages.Add($logTab)

$presetGroup = New-Object System.Windows.Forms.GroupBox
$presetGroup.Text = "Global Presets"
$presetGroup.Location = New-Object System.Drawing.Point(16, 14)
$presetGroup.Size = New-Object System.Drawing.Size(920, 145)
$generalTab.Controls.Add($presetGroup)

$coreGroup = New-Object System.Windows.Forms.GroupBox
$coreGroup.Text = "Run Setup"
$coreGroup.Location = New-Object System.Drawing.Point(16, 170)
$coreGroup.Size = New-Object System.Drawing.Size(465, 430)
$generalTab.Controls.Add($coreGroup)

$exportGroup = New-Object System.Windows.Forms.GroupBox
$exportGroup.Text = "Final Export"
$exportGroup.Location = New-Object System.Drawing.Point(495, 170)
$exportGroup.Size = New-Object System.Drawing.Size(440, 500)
$generalTab.Controls.Add($exportGroup)

$diagnosticsGroup = New-Object System.Windows.Forms.GroupBox
$diagnosticsGroup.Text = "Stage Diagnostics"
$diagnosticsGroup.Location = New-Object System.Drawing.Point(16, 615)
$diagnosticsGroup.Size = New-Object System.Drawing.Size(465, 360)
$generalTab.Controls.Add($diagnosticsGroup)

$exportTab = $exportGroup
$diagnosticsTab = $diagnosticsGroup

$presetLabel = New-Object System.Windows.Forms.Label
$presetLabel.Text = "Presets"
$presetLabel.Location = New-Object System.Drawing.Point(12, 22)
$presetLabel.Size = New-Object System.Drawing.Size(160, 20)
$presetGroup.Controls.Add($presetLabel)

$presetList = New-Object System.Windows.Forms.ListBox
$presetList.Location = New-Object System.Drawing.Point(12, 45)
$presetList.Size = New-Object System.Drawing.Size(745, 82)
foreach ($preset in $presets) { [void]$presetList.Items.Add($preset.Name + " - " + $preset.Note) }
$presetList.SelectedIndex = 0
$presetGroup.Controls.Add($presetList)
$ToolTip.SetToolTip($presetList, "Applies a complete known-good setup: GPU, resolution, samplers, export cleanup, diagnostics, and viewer behavior. You can still edit fields afterward.")

$applyPreset = New-Object System.Windows.Forms.Button
$applyPreset.Text = "Apply Preset"
$applyPreset.Location = New-Object System.Drawing.Point(775, 45)
$applyPreset.Size = New-Object System.Drawing.Size(115, 30)
$presetGroup.Controls.Add($applyPreset)

$gpuCombo = Add-Combo $coreGroup "GPU" 12 28 @("0 - Quadro RTX 5000", "1 - GeForce RTX 3070") "0 - Quadro RTX 5000" 220 "CUDA device to expose to TRELLIS. With CUDA_DEVICE_ORDER=PCI_BUS_ID, 0 is the RTX 5000 and 1 is the 3070 on this system."
$pipelineCombo = Add-Combo $coreGroup "Pipeline type" 12 63 @("512", "1024", "1024_cascade", "1536_cascade") "512" 160 "Output resolution path. The official app maps 512 to 512, 1024 to 1024_cascade, and 1536 to 1536_cascade."
$seedBox = Add-Number $coreGroup "Seed" 12 98 0 0 2147483647 0 1 "Deterministic seed for generation. Change it to try another result with the same settings."
$maxTokensBox = Add-Number $coreGroup "Max tokens" 12 133 12288 1024 200000 0 1024 "Caps high-resolution sparse latent token count during cascade upsampling. Lower values reduce memory but can reduce detail/resolution."
$modelBox = Add-Text $coreGroup "Model id/folder" 12 168 "microsoft/TRELLIS.2-4B" 175 "Hugging Face model id or a local folder containing pipeline.json and model weights."

$browseModel = New-Object System.Windows.Forms.Button
$browseModel.Text = "Browse..."
$browseModel.Location = New-Object System.Drawing.Point(375, 166)
$browseModel.Size = New-Object System.Drawing.Size(85, 28)
$browseModel.Add_Click({
    $folder = Pick-ModelFolder
    if ($folder) { $modelBox.Text = $folder }
})
$coreGroup.Controls.Add($browseModel)

$dtypeCombo = Add-Combo $coreGroup "Inference dtype" 12 218 @("auto", "fp16", "bf16", "float32") "auto" 160 "Precision for flow model inference. Use auto for this machine: it forces fp16 on CUDA because xformers sparse attention rejects bf16 on these RTX cards."
$bgCombo = Add-Combo $coreGroup "Background" 12 253 @("local-white", "keep", "trellis") "local-white" 160 "local-white removes near-white pixels locally; keep does no removal; trellis uses the configured RMBG model."
$whiteBox = Add-Number $coreGroup "White threshold" 12 288 245 0 255 0 1 "For local-white removal: pixels with R/G/B above this value become transparent. Lower removes more background but can erase pale objects."
$rembgBox = Add-Text $coreGroup "RMBG model" 12 323 "camenduru/RMBG-2.0:onnx/model_quantized.onnx" 230 "Hugging Face model id used when Background is trellis. Default is an ungated RMBG-2.0 ONNX mirror."

$stageDefaults = @{
    ss = @{ steps=12; guidance=7.5; rescale=0.7; rescaleT=5.0; intervalStart=0.6; intervalEnd=1.0 }
    shape = @{ steps=12; guidance=7.5; rescale=0.5; rescaleT=3.0; intervalStart=0.6; intervalEnd=1.0 }
    tex = @{ steps=12; guidance=1.0; rescale=0.0; rescaleT=3.0; intervalStart=0.6; intervalEnd=0.9 }
}
$ss = Add-StageControls $stagesTab "Stage 1: Sparse Structure" 20 25 $stageDefaults.ss
$shape = Add-StageControls $stagesTab "Stage 2: Shape Latent" 335 25 $stageDefaults.shape
$tex = Add-StageControls $stagesTab "Stage 3: Texture / Material Latent" 650 25 $stageDefaults.tex
$sparseResBox = Add-Number $ss.group "Sparse resolution" 12 210 0 0 128 0 4 "Stage-1 sparse structure grid resolution. 0 uses TRELLIS defaults: 32 for 512/cascade, 64 for direct 1024. Higher may cost substantially more VRAM."

$stageHelp = New-Object System.Windows.Forms.TextBox
$stageHelp.Multiline = $true
$stageHelp.ReadOnly = $true
$stageHelp.ScrollBars = "Vertical"
$stageHelp.Location = New-Object System.Drawing.Point(20, 295)
$stageHelp.Size = New-Object System.Drawing.Size(920, 300)
$stageHelp.Text = @"
Upstream app defaults:
  Sparse structure: steps 12, guidance 7.5, guidance rescale 0.7, rescale T 5.0, interval 0.6-1.0
  Shape latent:     steps 12, guidance 7.5, guidance rescale 0.5, rescale T 3.0, interval 0.6-1.0
  Texture latent:   steps 12, guidance 1.0, guidance rescale 0.0, rescale T 3.0, interval 0.6-0.9

Practical notes:
  - Steps are the biggest quality/runtime knob. The previous rough rocket was steps=1.
  - Guidance strength controls image adherence. Too much can create brittle artifacts.
  - Guidance rescale reduces over-guidance artifacts.
  - Guidance start/end limits where CFG is active during each sampling schedule.
  - 1024_cascade is the official app default quality path for resolution=1024.
  - 512 is much easier on VRAM and useful for smoke tests.
  - The PixelArtistry preset copies the Comfy advanced-generator values that exist in this local runner.
"@
$stagesTab.Controls.Add($stageHelp)

$decimationBox = Add-Number $exportTab "Decimation target" 12 28 300000 100000 5000000 0 10000 "Target face count during GLB extraction. Higher preserves detail but creates larger/slower assets. Try 1000000+ for characters."
$textureBox = Add-Number $exportTab "Texture size" 12 63 1024 1024 4096 0 1024 "Texture atlas size. Official app supports 1024, 2048, 4096; higher costs memory and time."
$outDirBox = Add-Text $exportTab "Output folder" 12 98 (Join-Path $Root "tmp") 155 "Folder where GLBs are written."

$browseOut = New-Object System.Windows.Forms.Button
$browseOut.Text = "Browse..."
$browseOut.Location = New-Object System.Drawing.Point(348, 96)
$browseOut.Size = New-Object System.Drawing.Size(85, 28)
$browseOut.Add_Click({
    $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
    $dialog.Description = "Choose output folder"
    if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $outDirBox.Text = $dialog.SelectedPath }
})
$exportTab.Controls.Add($browseOut)

$openViewer = New-Object System.Windows.Forms.CheckBox
$openViewer.Text = "Open GLB viewer when done"
$openViewer.AutoSize = $true
$openViewer.Checked = $true
$openViewer.Location = New-Object System.Drawing.Point(12, 143)
$ToolTip.SetToolTip($openViewer, "Starts a local HTTP server on 127.0.0.1:8765 and opens tools/glb_viewer.html with the generated GLB.")
$exportTab.Controls.Add($openViewer)

$noExport = New-Object System.Windows.Forms.CheckBox
$noExport.Text = "Run generation only, skip GLB export"
$noExport.AutoSize = $true
$noExport.Checked = $false
$noExport.Location = New-Object System.Drawing.Point(12, 170)
$ToolTip.SetToolTip($noExport, "Useful for debugging inference before the slower remesh/UV/GLB extraction step.")
$exportTab.Controls.Add($noExport)

$fillHoles = New-Object System.Windows.Forms.CheckBox
$fillHoles.Text = "Fill mesh holes during decode"
$fillHoles.AutoSize = $true
$fillHoles.Checked = $true
$fillHoles.Location = New-Object System.Drawing.Point(12, 197)
$ToolTip.SetToolTip($fillHoles, "Runs TRELLIS mesh hole filling before voxel/textured GLB export. Disabling can reveal raw geometry but usually worsens assets.")
$exportTab.Controls.Add($fillHoles)

$holePerimeterBox = Add-Number $exportTab "Max hole perimeter" 12 232 0.03 0.0001 1 4 0.001 "Maximum hole perimeter passed to TRELLIS fill_holes. Larger closes bigger holes but may bridge details incorrectly."

$exportRemesh = New-Object System.Windows.Forms.CheckBox
$exportRemesh.Text = "Remesh during GLB export"
$exportRemesh.AutoSize = $true
$exportRemesh.Checked = $true
$exportRemesh.Location = New-Object System.Drawing.Point(12, 265)
$ToolTip.SetToolTip($exportRemesh, "Runs o-voxel dual-contouring remesh before UV/texturing. Turning this off tests the non-remesh export path, but it may produce rougher topology.")
$exportTab.Controls.Add($exportRemesh)

$remeshBandBox = Add-Number $exportTab "Remesh band" 12 300 1.0 0.1 10 2 0.1 "Width of the narrow band used for remeshing. Smaller may preserve tighter features; larger can smooth/regularize more."
$remeshProjectBox = Add-Number $exportTab "Remesh project" 12 335 0.9 0 1 2 0.05 "Projection back to the original generated surface after remeshing. 0 is softer; 0.9 is o-voxel's sharper default."
$preSimplifyBox = Add-Number $exportTab "Pre-simplify target" 12 370 16777216 0 30000000 0 100000 "Initial mesh simplification before GLB export. 0 disables this pre-simplify test path."

$exportHelp = New-Object System.Windows.Forms.TextBox
$exportHelp.Multiline = $true
$exportHelp.ReadOnly = $true
$exportHelp.Location = New-Object System.Drawing.Point(12, 415)
$exportHelp.Size = New-Object System.Drawing.Size(410, 70)
$exportHelp.Text = @"
GLB extraction remeshes, simplifies, unwraps, samples voxel attributes, and writes a textured GLB.
If generation succeeds but export fails, try a lower decimation target or texture size.
For sharper tests, try remesh project 0.9, decimation 1000000+, and pre-simplify target 0.
The inline viewer is a convenience check; Blender remains the better inspection path for normals, topology, UVs, and materials.
"@
$exportTab.Controls.Add($exportHelp)

$saveStage1 = New-Object System.Windows.Forms.CheckBox
$saveStage1.Text = "Save Stage 1 sparse structure GLB"
$saveStage1.AutoSize = $true
$saveStage1.Checked = $true
$saveStage1.Location = New-Object System.Drawing.Point(12, 26)
$ToolTip.SetToolTip($saveStage1, "Exports active sparse-structure voxels as small blue boxes. This is an occupancy preview, not a real mesh.")
$diagnosticsTab.Controls.Add($saveStage1)

$saveStage2 = New-Object System.Windows.Forms.CheckBox
$saveStage2.Text = "Save Stage 2 shape-only GLB"
$saveStage2.AutoSize = $true
$saveStage2.Checked = $true
$saveStage2.Location = New-Object System.Drawing.Point(12, 56)
$ToolTip.SetToolTip($saveStage2, "Exports the decoded shape mesh before the texture/material stage. Use this to judge whether geometry is already soft.")
$diagnosticsTab.Controls.Add($saveStage2)

$saveStage3 = New-Object System.Windows.Forms.CheckBox
$saveStage3.Text = "Save Stage 3 decoded geometry GLB"
$saveStage3.AutoSize = $true
$saveStage3.Checked = $true
$saveStage3.Location = New-Object System.Drawing.Point(12, 86)
$ToolTip.SetToolTip($saveStage3, "Exports geometry after texture latent decode and optional hole filling, before o-voxel GLB remesh/UV/texture export.")
$diagnosticsTab.Controls.Add($saveStage3)

$script:openStageViewers = New-Object System.Windows.Forms.CheckBox
$script:openStageViewers.Text = "Open each stage artifact as soon as it is written"
$script:openStageViewers.AutoSize = $true
$script:openStageViewers.Checked = $true
$script:openStageViewers.Location = New-Object System.Drawing.Point(12, 116)
$ToolTip.SetToolTip($script:openStageViewers, "Opens a browser viewer for stage artifacts while generation continues. Disable if too many tabs are noisy.")
$diagnosticsTab.Controls.Add($script:openStageViewers)

$stopAfterCombo = Add-Combo $diagnosticsTab "Stop after stage" 12 154 @("none", "stage1", "stage2", "stage3") "none" 160 "Stops the run after the selected stage. Use stage1/2 to avoid spending time on later stages while diagnosing."
$stage1MaxBox = Add-Number $diagnosticsTab "Stage 1 voxel cap" 12 189 12000 1000 100000 0 1000 "Maximum number of sparse voxels exported as boxes for the stage-1 preview GLB. Higher is more complete but heavier to view."

$diagnosticsHelp = New-Object System.Windows.Forms.TextBox
$diagnosticsHelp.Multiline = $true
$diagnosticsHelp.ReadOnly = $true
$diagnosticsHelp.Location = New-Object System.Drawing.Point(12, 225)
$diagnosticsHelp.Size = New-Object System.Drawing.Size(425, 115)
$diagnosticsHelp.Text = @"
Stage artifacts:
  Stage 1 sparse: active occupancy voxels only. Good for silhouette/volume coverage.
  Stage 2 shape: decoded geometry before texture/material latent. Good for checking whether folds, face planes, and props exist geometrically.
  Stage 3 decoded: geometry after texture latent and hole filling, before final o-voxel remesh/UV/texture bake.
  Final GLB: full export path with remesh/simplification/UV/textures, controlled on the Export tab.

Diagnostic workflow:
  1. Stop after stage1 when checking background masks and gross silhouette.
  2. Stop after stage2 when checking geometry softness without texture/export variables.
  3. Run through stage3/final when checking texture bake and export cleanup.
"@
$diagnosticsTab.Controls.Add($diagnosticsHelp)

$faithcGroup = New-Object System.Windows.Forms.GroupBox
$faithcGroup.Text = "FaithC Contour Module"
$faithcGroup.Location = New-Object System.Drawing.Point(16, 16)
$faithcGroup.Size = New-Object System.Drawing.Size(450, 285)
$modulesTab.Controls.Add($faithcGroup)

$faithcModeCombo = Add-Combo $faithcGroup "Run FaithC" 12 30 @("off", "after-stage2", "after-stage3", "after-final") "off" 160 "Runs Faithful Contouring on a TRELLIS mesh artifact. Best use: compare contour/export fidelity after Stage 2, Stage 3, or final GLB."
$faithcResolutionBox = Add-Number $faithcGroup "Resolution" 12 65 256 32 2048 0 32 "FaithC contour grid resolution. Higher can preserve more geometry but can use much more VRAM."
$faithcTriCombo = Add-Combo $faithcGroup "Triangulation" 12 100 @("auto", "length", "angle", "normal_abs", "normal", "simple_02", "simple_13") "auto" 160 "How FaithC splits reconstructed quads into triangles."
$faithcMarginBox = Add-Number $faithcGroup "Normalize margin" 12 135 0.05 0 0.5 2 0.01 "Margin used only when normalization is enabled."
$faithcNormalize = New-Object System.Windows.Forms.CheckBox
$faithcNormalize.Text = "Normalize mesh into FaithC bounds"
$faithcNormalize.AutoSize = $true
$faithcNormalize.Checked = $false
$faithcNormalize.Location = New-Object System.Drawing.Point(12, 170)
$ToolTip.SetToolTip($faithcNormalize, "TRELLIS meshes are already small and centered; leave off first. Enable if FaithC reports zero active voxels.")
$faithcGroup.Controls.Add($faithcNormalize)
$faithcClamp = New-Object System.Windows.Forms.CheckBox
$faithcClamp.Text = "Clamp anchors"
$faithcClamp.AutoSize = $true
$faithcClamp.Checked = $true
$faithcClamp.Location = New-Object System.Drawing.Point(12, 198)
$ToolTip.SetToolTip($faithcClamp, "Clamps FaithC anchors to voxel bounds and projects them back to the surface.")
$faithcGroup.Controls.Add($faithcClamp)
$faithcFlux = New-Object System.Windows.Forms.CheckBox
$faithcFlux.Text = "Compute edge flux"
$faithcFlux.AutoSize = $true
$faithcFlux.Checked = $true
$faithcFlux.Location = New-Object System.Drawing.Point(12, 226)
$ToolTip.SetToolTip($faithcFlux, "Required for FaithC mesh reconstruction. Disable only for debugging token extraction.")
$faithcGroup.Controls.Add($faithcFlux)

$ultraGroup = New-Object System.Windows.Forms.GroupBox
$ultraGroup.Text = "UltraShape Refinement Module"
$ultraGroup.Location = New-Object System.Drawing.Point(485, 16)
$ultraGroup.Size = New-Object System.Drawing.Size(450, 390)
$modulesTab.Controls.Add($ultraGroup)

$ultraModeCombo = Add-Combo $ultraGroup "Run UltraShape" 12 30 @("off", "after-stage2", "after-stage3", "after-final") "off" 160 "Runs UltraShape on a TRELLIS coarse mesh artifact. Requires downloaded UltraShape checkpoint."
$ultraCkptBox = Add-Text $ultraGroup "Checkpoint" 12 65 (Join-Path $Root "integrations\UltraShape-1.0\checkpoints\ultrashape_v1.pt") 175 "Path to UltraShape checkpoint, e.g. ultrashape_v1.pt from the Hugging Face release."
$browseUltraCkpt = New-Object System.Windows.Forms.Button
$browseUltraCkpt.Text = "Browse..."
$browseUltraCkpt.Location = New-Object System.Drawing.Point(375, 63)
$browseUltraCkpt.Size = New-Object System.Drawing.Size(70, 28)
$browseUltraCkpt.Add_Click({
    $file = Pick-File "Choose UltraShape checkpoint" "PyTorch weights|*.pt;*.pth;*.ckpt|All files|*.*"
    if ($file) { $ultraCkptBox.Text = $file }
})
$ultraGroup.Controls.Add($browseUltraCkpt)
$ultraConfigBox = Add-Text $ultraGroup "Config" 12 100 (Join-Path $Root "integrations\UltraShape-1.0\configs\infer_dit_refine.yaml") 175 "Path to UltraShape inference config."
$browseUltraConfig = New-Object System.Windows.Forms.Button
$browseUltraConfig.Text = "Browse..."
$browseUltraConfig.Location = New-Object System.Drawing.Point(375, 98)
$browseUltraConfig.Size = New-Object System.Drawing.Size(70, 28)
$browseUltraConfig.Add_Click({
    $file = Pick-File "Choose UltraShape config" "YAML|*.yaml;*.yml|All files|*.*"
    if ($file) { $ultraConfigBox.Text = $file }
})
$ultraGroup.Controls.Add($browseUltraConfig)
$ultraStepsBox = Add-Number $ultraGroup "Steps" 12 140 12 1 200 0 1 "UltraShape DiT inference steps. README default is 50; 12 is the suggested speed test."
$ultraLatentsBox = Add-Number $ultraGroup "Num latents" 12 175 8192 1024 32768 0 512 "UltraShape latent token count. Lower is safer for VRAM; README suggests 8192 for low VRAM."
$ultraChunkBox = Add-Number $ultraGroup "Chunk size" 12 210 2048 512 10000 0 512 "UltraShape inference chunk size. Lower is slower but reduces VRAM."
$ultraOctreeBox = Add-Number $ultraGroup "Octree res" 12 245 512 64 2048 0 64 "UltraShape mesh extraction resolution. 1024 is default; 512 is a safer first test."
$ultraScaleBox = Add-Number $ultraGroup "Mesh scale" 12 280 0.99 0.1 2.0 2 0.01 "UltraShape mesh normalization scale."
$ultraLowVram = New-Object System.Windows.Forms.CheckBox
$ultraLowVram.Text = "UltraShape low VRAM"
$ultraLowVram.AutoSize = $true
$ultraLowVram.Checked = $true
$ultraLowVram.Location = New-Object System.Drawing.Point(12, 315)
$ToolTip.SetToolTip($ultraLowVram, "Uses UltraShape CPU offload mode when available.")
$ultraGroup.Controls.Add($ultraLowVram)
$ultraRemoveBg = New-Object System.Windows.Forms.CheckBox
$ultraRemoveBg.Text = "UltraShape remove background"
$ultraRemoveBg.AutoSize = $true
$ultraRemoveBg.Checked = $false
$ultraRemoveBg.Location = New-Object System.Drawing.Point(12, 343)
$ToolTip.SetToolTip($ultraRemoveBg, "Requests UltraShape's own background remover. Usually leave off when TRELLIS already prepared the image.")
$ultraGroup.Controls.Add($ultraRemoveBg)

$moduleHelp = New-Object System.Windows.Forms.TextBox
$moduleHelp.Multiline = $true
$moduleHelp.ReadOnly = $true
$moduleHelp.Location = New-Object System.Drawing.Point(16, 420)
$moduleHelp.Size = New-Object System.Drawing.Size(920, 130)
$moduleHelp.Text = @"
Post modules run after TRELLIS has emitted the selected artifact and released its model memory.
FaithC is an export/contouring comparison: it can preserve or expose mesh contour behavior but cannot invent missing Stage 2 geometry.
UltraShape is a learned mesh refinement pass: it needs a coarse mesh, the reference image, and a downloaded checkpoint.
Artifacts from these modules are printed as stage artifacts and opened by the same viewer toggle.
"@
$modulesTab.Controls.Add($moduleHelp)

$script:logBox = New-Object System.Windows.Forms.TextBox
$script:logBox.Multiline = $true
$script:logBox.ReadOnly = $true
$script:logBox.ScrollBars = "Both"
$script:logBox.WordWrap = $false
$script:logBox.Font = New-Object System.Drawing.Font("Consolas", 9)
$script:logBox.Dock = "Fill"
$logTab.Controls.Add($script:logBox)

function Set-Preset {
    if ($presetList.SelectedIndex -lt 0) { return }
    $preset = $presets[$presetList.SelectedIndex]
    $gpuCombo.SelectedIndex = if ($preset.Gpu -eq "0") { 0 } else { 1 }
    $pipelineCombo.SelectedItem = $preset.Pipeline
    $ss.steps.Value = $stageDefaults.ss.steps
    $ss.guidance.Value = $stageDefaults.ss.guidance
    $ss.rescale.Value = $stageDefaults.ss.rescale
    $ss.rescaleT.Value = $stageDefaults.ss.rescaleT
    $ss.intervalStart.Value = $stageDefaults.ss.intervalStart
    $ss.intervalEnd.Value = $stageDefaults.ss.intervalEnd
    $shape.steps.Value = $stageDefaults.shape.steps
    $shape.guidance.Value = $stageDefaults.shape.guidance
    $shape.rescale.Value = $stageDefaults.shape.rescale
    $shape.rescaleT.Value = $stageDefaults.shape.rescaleT
    $shape.intervalStart.Value = $stageDefaults.shape.intervalStart
    $shape.intervalEnd.Value = $stageDefaults.shape.intervalEnd
    $tex.steps.Value = $stageDefaults.tex.steps
    $tex.guidance.Value = $stageDefaults.tex.guidance
    $tex.rescale.Value = $stageDefaults.tex.rescale
    $tex.rescaleT.Value = $stageDefaults.tex.rescaleT
    $tex.intervalStart.Value = $stageDefaults.tex.intervalStart
    $tex.intervalEnd.Value = $stageDefaults.tex.intervalEnd
    if ($preset.PSObject.Properties['Steps']) {
        foreach ($stage in @($ss, $shape, $tex)) { $stage.steps.Value = $preset.Steps }
    }
    if ($preset.PSObject.Properties['SparseSteps']) { $ss.steps.Value = $preset.SparseSteps }
    if ($preset.PSObject.Properties['SparseGuidance']) { $ss.guidance.Value = $preset.SparseGuidance }
    if ($preset.PSObject.Properties['SparseRescale']) { $ss.rescale.Value = $preset.SparseRescale }
    if ($preset.PSObject.Properties['SparseRescaleT']) { $ss.rescaleT.Value = $preset.SparseRescaleT }
    if ($preset.PSObject.Properties['ShapeSteps']) { $shape.steps.Value = $preset.ShapeSteps }
    if ($preset.PSObject.Properties['ShapeGuidance']) { $shape.guidance.Value = $preset.ShapeGuidance }
    if ($preset.PSObject.Properties['ShapeRescale']) { $shape.rescale.Value = $preset.ShapeRescale }
    if ($preset.PSObject.Properties['ShapeRescaleT']) { $shape.rescaleT.Value = $preset.ShapeRescaleT }
    if ($preset.PSObject.Properties['TexSteps']) { $tex.steps.Value = $preset.TexSteps }
    if ($preset.PSObject.Properties['TexGuidance']) { $tex.guidance.Value = $preset.TexGuidance }
    if ($preset.PSObject.Properties['TexRescale']) { $tex.rescale.Value = $preset.TexRescale }
    if ($preset.PSObject.Properties['TexRescaleT']) { $tex.rescaleT.Value = $preset.TexRescaleT }
    if ($preset.PSObject.Properties['SparseStart']) { $ss.intervalStart.Value = $preset.SparseStart }
    if ($preset.PSObject.Properties['SparseEnd']) { $ss.intervalEnd.Value = $preset.SparseEnd }
    if ($preset.PSObject.Properties['ShapeStart']) { $shape.intervalStart.Value = $preset.ShapeStart }
    if ($preset.PSObject.Properties['ShapeEnd']) { $shape.intervalEnd.Value = $preset.ShapeEnd }
    if ($preset.PSObject.Properties['TexStart']) { $tex.intervalStart.Value = $preset.TexStart }
    if ($preset.PSObject.Properties['TexEnd']) { $tex.intervalEnd.Value = $preset.TexEnd }
    $maxTokensBox.Value = $preset.MaxTokens
    if ($preset.PSObject.Properties['SparseRes']) { $sparseResBox.Value = $preset.SparseRes } else { $sparseResBox.Value = 0 }
    $decimationBox.Value = $preset.Decimation
    $textureBox.Value = $preset.Texture
    if ($preset.PSObject.Properties['NoExport']) { $noExport.Checked = [bool]$preset.NoExport } else { $noExport.Checked = $false }
    if ($preset.PSObject.Properties['FillHoles']) { $fillHoles.Checked = [bool]$preset.FillHoles } else { $fillHoles.Checked = $true }
    if ($preset.PSObject.Properties['ExportRemesh']) { $exportRemesh.Checked = [bool]$preset.ExportRemesh } else { $exportRemesh.Checked = $true }
    if ($preset.PSObject.Properties['RemeshBand']) { $remeshBandBox.Value = [decimal]$preset.RemeshBand } else { $remeshBandBox.Value = 1.0 }
    if ($preset.PSObject.Properties['RemeshProject']) { $remeshProjectBox.Value = [decimal]$preset.RemeshProject } else { $remeshProjectBox.Value = 0.9 }
    if ($preset.PSObject.Properties['PreSimplify']) { $preSimplifyBox.Value = [decimal]$preset.PreSimplify } else { $preSimplifyBox.Value = 16777216 }
    if ($preset.PSObject.Properties['SaveStage1']) { $saveStage1.Checked = [bool]$preset.SaveStage1 } else { $saveStage1.Checked = $true }
    if ($preset.PSObject.Properties['SaveStage2']) { $saveStage2.Checked = [bool]$preset.SaveStage2 } else { $saveStage2.Checked = $true }
    if ($preset.PSObject.Properties['SaveStage3']) { $saveStage3.Checked = [bool]$preset.SaveStage3 } else { $saveStage3.Checked = $true }
    if ($preset.PSObject.Properties['OpenStageViewers']) { $script:openStageViewers.Checked = [bool]$preset.OpenStageViewers } else { $script:openStageViewers.Checked = $true }
    if ($preset.PSObject.Properties['StopAfter']) { $stopAfterCombo.SelectedItem = $preset.StopAfter } else { $stopAfterCombo.SelectedItem = "none" }
    if ($preset.PSObject.Properties['OpenViewer']) { $openViewer.Checked = [bool]$preset.OpenViewer } else { $openViewer.Checked = $true }
    if ($preset.PSObject.Properties['FaithCMode']) { $faithcModeCombo.SelectedItem = $preset.FaithCMode } else { $faithcModeCombo.SelectedItem = "off" }
    if ($preset.PSObject.Properties['FaithCResolution']) { $faithcResolutionBox.Value = [decimal]$preset.FaithCResolution } else { $faithcResolutionBox.Value = 256 }
    $faithcTriCombo.SelectedItem = "auto"
    $faithcNormalize.Checked = $false
    $faithcClamp.Checked = $true
    $faithcFlux.Checked = $true
    if ($preset.PSObject.Properties['UltraShapeMode']) { $ultraModeCombo.SelectedItem = $preset.UltraShapeMode } else { $ultraModeCombo.SelectedItem = "off" }
}
$applyPreset.Add_Click({ Set-Preset })
$presetList.Add_DoubleClick({ Set-Preset })
Set-Preset

$run = New-Object System.Windows.Forms.Button
$run.Text = "Choose Image and Run"
$run.Location = New-Object System.Drawing.Point(15, 680)
$run.Size = New-Object System.Drawing.Size(160, 34)
$run.Anchor = "Bottom,Left"
$form.Controls.Add($run)

$cancel = New-Object System.Windows.Forms.Button
$cancel.Text = "Close"
$cancel.Location = New-Object System.Drawing.Point(188, 680)
$cancel.Size = New-Object System.Drawing.Size(90, 34)
$cancel.Anchor = "Bottom,Left"
$cancel.Add_Click({
    if ($script:currentProcess -and -not $script:currentProcess.HasExited) {
        Append-Log "Stopping TRELLIS.2 process..."
        try {
            & taskkill.exe /PID $script:currentProcess.Id /T /F | Out-Null
            if ($script:runTimer) { $script:runTimer.Stop() }
            Set-RunningState $false
        } catch {
            Append-Log ("Failed to stop process: " + $_.Exception.Message)
        }
    } else {
        $form.Close()
    }
})
$form.Controls.Add($cancel)

$statusLabel = New-Object System.Windows.Forms.Label
$statusLabel.Text = "Idle"
$statusLabel.Location = New-Object System.Drawing.Point(300, 688)
$statusLabel.Size = New-Object System.Drawing.Size(650, 24)
$statusLabel.Anchor = "Bottom,Left,Right"
$form.Controls.Add($statusLabel)

$run.Add_Click({
    $image = Pick-InputImage
    if (-not $image) { return }

    $gpu = if ($gpuCombo.SelectedIndex -eq 0) { "0" } else { "1" }
    $pipeline = [string]$pipelineCombo.SelectedItem
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $safeName = [System.IO.Path]::GetFileNameWithoutExtension($image) -replace '[^A-Za-z0-9_.-]', '_'
    if (-not (Test-Path $outDirBox.Text)) { New-Item -ItemType Directory -Path $outDirBox.Text | Out-Null }
    $out = Join-Path $outDirBox.Text "${safeName}_${pipeline}_${stamp}.glb"

    $env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
    $env:CUDA_VISIBLE_DEVICES = $gpu
    $env:ATTN_BACKEND = "sdpa"
    $env:SPARSE_ATTN_BACKEND = "xformers"
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

    $args = @(
        "local_image_to_3d.py",
        $image,
        "--out", $out,
        "--model", $modelBox.Text,
        "--pipeline-type", $pipeline,
        "--seed", "$([int]$seedBox.Value)",
        "--max-num-tokens", "$([int]$maxTokensBox.Value)",
        "--decimation-target", "$([int]$decimationBox.Value)",
        "--texture-size", "$([int]$textureBox.Value)",
        "--remesh-band", "$([double]$remeshBandBox.Value)",
        "--remesh-project", "$([double]$remeshProjectBox.Value)",
        "--pre-simplify-target", "$([int]$preSimplifyBox.Value)",
        "--sparse-structure-resolution", "$([int]$sparseResBox.Value)",
        "--inference-dtype", ([string]$dtypeCombo.SelectedItem),
        "--background", ([string]$bgCombo.SelectedItem),
        "--rembg-model", $rembgBox.Text,
        "--white-threshold", "$([int]$whiteBox.Value)",
        "--ss-steps", "$([int]$ss.steps.Value)",
        "--ss-guidance-strength", "$([double]$ss.guidance.Value)",
        "--ss-guidance-rescale", "$([double]$ss.rescale.Value)",
        "--ss-rescale-t", "$([double]$ss.rescaleT.Value)",
        "--ss-guidance-interval-start", "$([double]$ss.intervalStart.Value)",
        "--ss-guidance-interval-end", "$([double]$ss.intervalEnd.Value)",
        "--shape-steps", "$([int]$shape.steps.Value)",
        "--shape-guidance-strength", "$([double]$shape.guidance.Value)",
        "--shape-guidance-rescale", "$([double]$shape.rescale.Value)",
        "--shape-rescale-t", "$([double]$shape.rescaleT.Value)",
        "--shape-guidance-interval-start", "$([double]$shape.intervalStart.Value)",
        "--shape-guidance-interval-end", "$([double]$shape.intervalEnd.Value)",
        "--tex-steps", "$([int]$tex.steps.Value)",
        "--tex-guidance-strength", "$([double]$tex.guidance.Value)",
        "--tex-guidance-rescale", "$([double]$tex.rescale.Value)",
        "--tex-rescale-t", "$([double]$tex.rescaleT.Value)",
        "--tex-guidance-interval-start", "$([double]$tex.intervalStart.Value)",
        "--tex-guidance-interval-end", "$([double]$tex.intervalEnd.Value)",
        "--max-hole-perimeter", "$([double]$holePerimeterBox.Value)",
        "--stop-after", ([string]$stopAfterCombo.SelectedItem),
        "--stage1-max-voxels", "$([int]$stage1MaxBox.Value)",
        "--faithc-mode", ([string]$faithcModeCombo.SelectedItem),
        "--faithc-resolution", "$([int]$faithcResolutionBox.Value)",
        "--faithc-tri-mode", ([string]$faithcTriCombo.SelectedItem),
        "--faithc-margin", "$([double]$faithcMarginBox.Value)",
        "--ultrashape-mode", ([string]$ultraModeCombo.SelectedItem),
        "--ultrashape-config", $ultraConfigBox.Text,
        "--ultrashape-ckpt", $ultraCkptBox.Text,
        "--ultrashape-steps", "$([int]$ultraStepsBox.Value)",
        "--ultrashape-scale", "$([double]$ultraScaleBox.Value)",
        "--ultrashape-num-latents", "$([int]$ultraLatentsBox.Value)",
        "--ultrashape-chunk-size", "$([int]$ultraChunkBox.Value)",
        "--ultrashape-octree-res", "$([int]$ultraOctreeBox.Value)"
    )
    if ($noExport.Checked) { $args += "--no-export" }
    if (-not $fillHoles.Checked) { $args += "--no-fill-holes" }
    if (-not $exportRemesh.Checked) { $args += "--no-export-remesh" }
    if ($saveStage1.Checked) { $args += "--save-stage1" }
    if ($saveStage2.Checked) { $args += "--save-stage2" }
    if ($saveStage3.Checked) { $args += "--save-stage3" }
    if ($faithcNormalize.Checked) { $args += "--faithc-normalize" } else { $args += "--no-faithc-normalize" }
    if ($faithcClamp.Checked) { $args += "--faithc-clamp-anchors" } else { $args += "--no-faithc-clamp-anchors" }
    if ($faithcFlux.Checked) { $args += "--faithc-compute-flux" } else { $args += "--no-faithc-compute-flux" }
    if ($ultraLowVram.Checked) { $args += "--ultrashape-low-vram" } else { $args += "--no-ultrashape-low-vram" }
    if ($ultraRemoveBg.Checked) { $args += "--ultrashape-remove-bg" }

    $script:logBox.Clear()
    $script:openedArtifacts = @{}
    $tabs.SelectedTab = $logTab
    Append-Log "Running TRELLIS.2"
    Append-Log "Input:    $image"
    Append-Log "Output:   $out"
    Append-Log "GPU:      $gpu"
    Append-Log "Pipeline: $pipeline"
    Append-Log ""

    $runDir = Join-Path $Root "tmp\launcher-runs"
    if (-not (Test-Path $runDir)) { New-Item -ItemType Directory -Path $runDir | Out-Null }
    $runId = Get-Date -Format "yyyyMMdd_HHmmss_ffff"
    $cmdPath = Join-Path $runDir "trellis2_$runId.cmd"
    $logPath = Join-Path $runDir "trellis2_$runId.log"
    Set-Content -Path $logPath -Value "" -Encoding ASCII

    $pythonExe = Join-Path $Root ".venv\Scripts\python.exe"
    $pythonArgs = ($args | ForEach-Object { [string](Quote-CommandArgument ([string]$_)) }) -join " "
    $cmdLines = New-Object System.Collections.Generic.List[string]
    $cmdLines.Add("@echo off")
    $cmdLines.Add("set CUDA_DEVICE_ORDER=PCI_BUS_ID")
    $cmdLines.Add("set CUDA_VISIBLE_DEVICES=$gpu")
    $cmdLines.Add("set ATTN_BACKEND=sdpa")
    $cmdLines.Add("set SPARSE_ATTN_BACKEND=xformers")
    $cmdLines.Add("set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    $cmdLines.Add([string]("set PATH=" + (Join-Path $Root ".venv\Scripts") + ";%PATH%"))
    $cmdLines.Add([string]("cd /d " + (Quote-CommandArgument $Root)))
    $cmdLines.Add([string]((Quote-CommandArgument $pythonExe) + " -u " + $pythonArgs + " > " + (Quote-CommandArgument $logPath) + " 2>&1"))
    $cmdLines.Add("exit /b %ERRORLEVEL%")
    Set-Content -Path $cmdPath -Value $cmdLines -Encoding ASCII

    $script:currentProcess = $null
    $script:currentOutput = $out
    $script:currentNoExport = $noExport.Checked
    $script:currentOpenViewer = $openViewer.Checked
    $script:currentLog = $logPath
    $script:logReadOffset = 0

    Set-RunningState $true
    try {
        $cmdArgument = '/c ""' + $cmdPath + '""'
        $script:currentProcess = Start-Process -FilePath "cmd.exe" -ArgumentList $cmdArgument -WorkingDirectory $Root -WindowStyle Hidden -PassThru
        if ($script:runTimer) { $script:runTimer.Stop() }
        $script:runTimer = New-Object System.Windows.Forms.Timer
        $script:runTimer.Interval = 1000
        $script:runTimer.Add_Tick({
            Update-RunLog
            if ($script:currentProcess -and $script:currentProcess.HasExited) {
                $script:runTimer.Stop()
                Update-RunLog
                $exitCode = $script:currentProcess.ExitCode
                Set-RunningState $false
                Append-Log ""
                if ($exitCode -eq 0 -and -not $script:currentNoExport -and (Test-Path $script:currentOutput)) {
                    Append-Log "Done: $script:currentOutput"
                    if ($script:currentOpenViewer) { Start-Viewer -GlbPath $script:currentOutput }
                } elseif ($exitCode -eq 0) {
                    Append-Log "Generation completed."
                } else {
                    Append-Log "TRELLIS.2 failed with exit code $exitCode"
                }
                $script:currentProcess = $null
            }
        })
        $script:runTimer.Start()
    } catch {
        Set-RunningState $false
        Append-Log ("Failed to start TRELLIS.2: " + $_.Exception.Message)
        $script:currentProcess = $null
    }
})

[void]$form.ShowDialog()
