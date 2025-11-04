# Deployment guide# Deployment Guide

## Local Development

1. **Prerequisites**
   - Docker & Docker-Compose
   - Python 3.10+
   - GCP Service Account Key with Storage Admin permissions

2. **Setup**
   ```bash
   git clone <repository-url>
   cd crop-monitor
   cp .env.example .env
   # Edit .env with your local values
   docker-compose up -d