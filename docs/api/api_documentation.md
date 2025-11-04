
**File: `docs/api/api_documentation.md`**
```markdown
# API Documentation

The primary API is a FastAPI application. Interactive Swagger/OpenAPI docs are available at `/docs` when the server is running.

## Base URL
- Local: `http://localhost:8080`
- Production: `https://your-production-url.com`

## Authentication
Most endpoints require a Firebase JWT token sent in the `Authorization` header as a Bearer token.
`Authorization: Bearer <your_token>`

## Key Endpoints

### Health Check
**GET /health**
- Checks API connectivity and database status.
- Response: `{"status": "ok", "timestamp": "2023-11-07T10:00:00Z"}`

### Image Prediction
**POST /predict**
- Upload an image for disease detection.
- **Body**: `multipart/form-data` with a file field `image`.
- **Response**:
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "message": "Inference started successfully."
}