$files = @(
    'src/silentfail/detectors.py',
    'src/silentfail/runner.py',
    'src/silentfail/tools.py',
    'tests/test_model_proposals.py',
    'tests/test_pipeline_artifact_sync.py'
)
foreach ($f in $files) {
    $hits = Select-String -Pattern '<<<<<<<','=======','>>>>>>>' -Path $f -SimpleMatch
    if ($hits) {
        foreach ($h in $hits) {
            Write-Output ('{0}:{1}' -f $f, $h.LineNumber)
        }
    } else {
        Write-Output ('{0}: CLEAN' -f $f)
    }
}
