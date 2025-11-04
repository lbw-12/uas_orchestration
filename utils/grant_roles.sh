# Grant your user the ability to view project resources
gcloud projects add-iam-policy-binding uas-orchestration-engine \
    --member="user:lwaltz12@gmail.com" \
    --role="roles/viewer"

# Grant your user the ability to view GCS buckets and objects
gcloud projects add-iam-policy-binding uas-orchestration-engine \
    --member="user:lwaltz12@gmail.com" \
    --role="roles/storage.objectViewer"