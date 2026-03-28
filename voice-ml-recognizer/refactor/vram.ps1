# Get-vRAM.ps1

Write-Host "`n=== GPU vRAM Usage by Process ===" -ForegroundColor Cyan
Write-Host "Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n" -ForegroundColor Gray

try {
    $gpuMemCounters = Get-Counter "\GPU Process Memory(*)\Dedicated Usage" -ErrorAction Stop

    $results = $gpuMemCounters.CounterSamples |
        Where-Object { $_.CookedValue -gt 0 } |
        ForEach-Object {
            if ($_.InstanceName -match "pid_(\d+)") {
                $procId = [int]$Matches[1]   # <-- fix: $pid è riservato in PS
                $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
                [PSCustomObject]@{
                    PID        = $procId
                    Name       = if ($proc) { $proc.Name } else { "<exited>" }
                    vRAM_MB    = [math]::Round($_.CookedValue / 1MB, 2)
                    vRAM_Bytes = $_.CookedValue
                }
            }
        } |
        Where-Object { $_ -ne $null } |
        Sort-Object vRAM_Bytes -Descending

    if ($results) {
        $results | Format-Table -AutoSize -Property PID, Name, @{
            Label      = "vRAM (MB)"
            Expression = { $_.vRAM_MB }
            Align      = "Right"
        }

        $totalMB = [math]::Round(($results | Measure-Object -Property vRAM_Bytes -Sum).Sum / 1MB, 2)
        Write-Host "Total Dedicated vRAM in use: $totalMB MB" -ForegroundColor Yellow
    } else {
        Write-Host "Nessun processo con vRAM dedicata trovato." -ForegroundColor Yellow
    }
}
catch {
    Write-Warning "Errore: $_"
}

Write-Host "`n=== GPU Adapters (WMI) ===" -ForegroundColor Cyan
Get-WmiObject Win32_VideoController |
    Select-Object Name,
        @{ N="VRAM Total (MB)"; E={ [math]::Round($_.AdapterRAM / 1MB, 0) } },
        DriverVersion, VideoProcessor |
    Format-Table -AutoSize