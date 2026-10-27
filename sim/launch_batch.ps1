# Launch sim/run_batch.m on N parallel Octave workers, then merge the QC files.
#   powershell -File sim\launch_batch.ps1 -First 1 -Last 20 -Workers 8
#   powershell -File sim\launch_batch.ps1 -First 1 -Last 3150 -Workers 8      (full dataset)
param(
    [int]$First = 1,
    [int]$Last = 20,
    [int]$Workers = 8,
    [string]$OutDir = 'D:/USV-landing-prediction/data/sim/dataset_v1',
    [string]$Octave = 'D:\tools\octave\mingw64\bin\octave-cli.exe'
)
$sim = 'D:/USV-landing-prediction/sim'
$design = "$sim/design_matrix.csv"
$logDir = Join-Path ($OutDir -replace '/', '\') 'logs'
New-Item -ItemType Directory -Force $logDir | Out-Null
Copy-Item ($design -replace '/', '\') (Join-Path ($OutDir -replace '/', '\') 'design_matrix.csv') -Force

$t0 = Get-Date
$procs = foreach ($k in 1..$Workers) {
    $eval = "cd('$sim'); mss_setup; run_batch('$design', '$OutDir', $First, $Last, $k, $Workers);"
    Start-Process -FilePath $Octave -NoNewWindow -PassThru `
        -RedirectStandardOutput (Join-Path $logDir "worker$k.out.txt") `
        -RedirectStandardError (Join-Path $logDir "worker$k.err.txt") `
        -ArgumentList '--no-gui', '--quiet', '--eval', "`"$eval`""
}
$procs | Wait-Process
"batch ids $First..$Last on $Workers workers: {0:N0} s" -f ((Get-Date) - $t0).TotalSeconds
& python "$sim/batch_report.py" $OutDir
