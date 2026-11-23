# Merge split parts back into yolo26x.pt
$ErrorActionPreference = "Stop"
$dir = $PSScriptRoot
$parts = Get-ChildItem $dir -Filter "yolo26x.pt.part*" | Sort-Object Name
if ($parts.Count -lt 1) { throw "未找到 yolo26x.pt.part*" }
$out = Join-Path $dir "yolo26x.pt"
$dest = [System.IO.File]::Create($out)
try {
  foreach ($p in $parts) {
    Write-Host "写入 $($p.Name) ..."
    $bytes = [System.IO.File]::ReadAllBytes($p.FullName)
    $dest.Write($bytes, 0, $bytes.Length)
  }
} finally { $dest.Close() }
$size = (Get-Item $out).Length
Write-Host "完成: $out ($([math]::Round($size/1MB,2)) MB)"
if ($size -ne 118667365) { Write-Warning "大小不是预期的 118667365 字节" }