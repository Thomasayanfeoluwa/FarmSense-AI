
**File: `docs/architecture/system_architecture.md`**
```markdown
# System Architecture

## High-Level Overview

```mermaid
graph TB
    A[Farmer Devices<br/>Phone/Webcam/Drone] --> B[Frontend<br/>Streamlit/Next.js];
    B -- HTTP REST/Upload --> C[Backend API<br/>FastAPI];
    C --> D[Auth<br/>Firebase];
    C --> E[Databases<br/>PostgreSQL<br/>MongoDB];
    C -- Async Task --> F[Celery Worker<br/>Redis Broker];
    F --> G[AI Models<br/>PyTorch/TensorFlow Lite];
    G --> H[Cloud Storage<br/>GCP];
    C --> I[External APIs<br/>OpenWeatherMap<br/>Earth Engine];
    C --> J[Notifications<br/>Twilio/Gmail];
    F --> E;