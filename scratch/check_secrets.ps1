$untracked = @(
    'data/g1_live_evaluation_results.jsonl',
    'data/g1_partial_refund_cohort.jsonl',
    'data/pilot_pipeline_run.json',
    'data/run_g1_eval.py',
    'data/run_g1_partial_cohort.py',
    'tests/test_deterministic_retry_timeout.py',
    'tests/test_duplicate_webhook.py',
    'tests/test_duplicate_webhook_refund_id.py',
    'tests/test_empty200_keyless_retry.py',
    'tests/test_nebius_gate_g0_fallback.py',
    'tests/test_timeout_after_success_keyless_duplicate.py'
)
foreach ($f in $untracked) {
    Write-Output "=== $f ==="
    $hits = Select-String -Pattern 'api_key','API_KEY','secret','password','token','NEBIUS_API','sk-','Bearer' -Path $f -SimpleMatch
    if ($hits) {
        foreach ($h in $hits) {
            Write-Output ("  LINE {0}: {1}" -f $h.LineNumber, $h.Line.Trim())
        }
    } else {
        Write-Output "  No secrets found"
    }
}
