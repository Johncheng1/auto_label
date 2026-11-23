# 示例：对 images 目录做 YOLO26 辅助标注（LabelImg VOC XML）
# 用法: .\run_example.ps1 -Source "D:\data\images" -Model "yolo26n.pt"

param(
  [Parameter(Mandatory = $true)][string]$Source,
  [string]$Model = "yolo26n.pt",
  [string]$Format = "voc",
  [double]$Conf = 0.25,
  [string]$Out = ""
)

$script = Join-Path $PSScriptRoot "auto_label.py"
$argsList = @(
  $script,
  "--source", $Source,
  "--model", $Model,
  "--format", $Format,
  "--conf", $Conf
)
if ($Out) { $argsList += @("--out", $Out) }

python @argsList
