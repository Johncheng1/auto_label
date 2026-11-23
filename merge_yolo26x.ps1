# Merge split parts back into yolo26x.pt
# Usage: put this script next to yolo26x.pt.part01/02/03, then run:
#   powershell -ExecutionPolicy Bypass -File .\merge_yolo26x.ps1

$ErrorActionPreference = "Stop"
$dir = $PSScriptRoot
$parts = @(
  (Join-Path $dir "yolo26x.pt.part01"),
  (Join-Path $dir "yolo26x.pt.part02"),
  (Join-Path $dir "yolo26x.pt.part03")
)
$out = Join-Path $dir "yolo26x.pt"

foreach ($p in $parts) {
  if (-not (Test-Path $p)) { throw "缺少分卷: $p" }
}

$dest = [System.IO.File]::Create($out)
try {
  foreach ($p in $parts) {
    Write-Host "写入 $(Split-Path $p -Leaf) ..."
    $bytes = [System.IO.File]::ReadAllBytes($p)
    $dest.Write($bytes, 0, $bytes.Length)
  }
} finally {
  $dest.Close()
}

$size = (Get-Item $out).Length
Write-Host "完成: $out ($([math]::Round($size/1MB,2)) MB)"
if ($size -ne 118667365) {
  Write-Warning "大小不是预期的 118667365 字节，请检查分卷是否完整。"
}
