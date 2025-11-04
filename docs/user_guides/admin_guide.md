# Administrator guide# Administrator's Guide

## System Overview

This platform is a distributed system comprising:
- **Frontend**: Streamlit (MVP) and Next.js (Production) applications.
- **Backend**: FastAPI application handling requests and business logic.
- **AI Workers**: Celery workers for async model inference and training.
- **Databases**: PostgreSQL (structured data), MongoDB (unstructured data, logs).
- **Storage**: GCP Cloud Storage for images and models.
- **Monitoring**: Prometheus/Grafana for metrics, Sentry for errors.

## User Management

Admins can manage users through the Firebase Auth console or a custom admin panel (`/admin` route).
- Roles: `farmer`, `agronomist`, `admin`, `finance`.
- Permissions are enforced via API middleware.

## Model Management

1. **Deploying New Models**
   - Train new models in the `notebooks/` directory.
   - Export the model to `models/` and update the model version in `src/config/settings.py`.
   - The CI/CD pipeline will automatically test and deploy the new model to staging.

2. **Monitoring Model Performance**
   - Key metrics (Accuracy, F1-Score, Latency) are tracked in Prometheus.
   - Drift detection scripts run weekly (`src/utils/drift_detection.py`).
   - Review feedback logs in MongoDB `feedback` collection.

## System Maintenance

- **Logs**: Check Docker container logs: `docker-compose logs -f backend`
- **Database Backups**: Automated daily backups are configured in `deployment/scripts/backup.sh`.
- **Scaling**: Adjust the number of Celery worker replicas in `docker-compose.prod.yml`.