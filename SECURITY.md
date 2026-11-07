# Security policy

This research platform is not a safety system and must not control aircraft or industrial equipment. Please report vulnerabilities privately to the repository owner rather than opening a public exploit issue.

Never expose the development Compose defaults to the internet. For any shared environment, set `APP_ENV=production`, replace all default database credentials, require a strong `API_KEY`, restrict CORS, terminate TLS, isolate MLflow, and use a secret manager. Raw telemetry, model artifacts, audit records, and experiment metadata may be sensitive even when this reference dataset is public simulated data.

The HTTP retraining route validates dry runs only; promotion remains a deliberate CLI/operator action. Static Pages mode is read-only and contains only precomputed NASA benchmark results with an explicit offline label.
