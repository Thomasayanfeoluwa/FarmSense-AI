
**File: `docs/architecture/data_flow.md`**
```markdown
# Data Flow Documentation

## Image Upload & Inference Flow

```mermaid
sequenceDiagram
    participant Farmer
    participant Frontend
    participant API
    participant Redis
    participant Celery
    participant GCS
    participant MongoDB
    participant Model

    Farmer->>Frontend: Uploads Image
    Frontend->>API: POST /predict (with image)
    API->>API: Validate input, generate task ID
    API->>Redis: Add inference task to queue
    API->>Frontend: Return task ID: 202 Accepted
    Note over Frontend, API: Async processing begins

    loop Poll for result
        Frontend->>API: GET /result/{task_id}
        API->>MongoDB: Check for result
        MongoDB-->>API: Result (if available)
        API-->>Frontend: Return result or "PENDING"
    end

    Celery->>Redis: Fetch next task
    Celery->>GCS: Download image
    Celery->>Model: Run inference
    Model-->>Celery: Return prediction
    Celery->>MongoDB: Store prediction & metadata
    Celery->>GCS: Upload processed image (if any)