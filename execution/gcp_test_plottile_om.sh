job_base="plottile-om-wooster-replant-ms-20250430"
timestamp=$(date +%Y%m%d-%H%M%S)
job="${job_base}-${timestamp}"
echo "Submitting GCP Batch job: $job"

gcloud batch jobs submit $job \
    --config=../shell_scripts/generated_gcp/3_gcp_plottile_om/wooster_replant_ms_20250430.json \
    --project=uas-orchestration-engine \
    --location=us-central1
