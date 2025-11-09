# Docker Guide for ad3.py

## Quick Start

### Option 1: Using Docker directly

**Build the image:**
```bash
docker build -t case-closed-agent .
```

**Run the container (with 1 CPU limit for competition):**
```bash
docker run -d -p 5008:5008 --cpus="1.0" --name agent case-closed-agent
```

**Run with custom port:**
```bash
docker run -d -p 5009:5009 -e PORT=5009 --name agent case-closed-agent
```

**View logs:**
```bash
docker logs -f agent
```

**Stop the container:**
```bash
docker stop agent
docker rm agent
```

### Option 2: Using Docker Compose

**Start the agent:**
```bash
docker-compose up -d
```

**View logs:**
```bash
docker-compose logs -f
```

**Stop the agent:**
```bash
docker-compose down
```

## Testing

Once the container is running, test it:

```bash
# Test health endpoint
curl http://localhost:5008/

# Expected response:
# {"participant":"GeminiAI_Agent","agent_name":"CaseClosed_FinalBoss_v8_Fixed"}
```

## Configuration

### CPU Requirements

**`ad3.py` needs only 1 CPU** because:
- Flask runs single-threaded by default
- Minimax search is sequential (not parallelized)
- No multiprocessing or threading for parallel computation

The competition limit is 1 CPU, which matches the agent's architecture perfectly.

### Environment Variables

- `PORT`: Port to run the Flask app on (default: 5008)

**Example:**
```bash
docker run -d -p 5009:5009 -e PORT=5009 case-closed-agent
```

### Port Mapping

The format is `host_port:container_port`:
- `-p 5008:5008` maps host port 5008 to container port 5008
- `-p 5009:5008` maps host port 5009 to container port 5008

## Building for Production

**Multi-stage build (smaller image):**
```dockerfile
FROM python:3.12-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY ad3.py .
ENV PATH=/root/.local/bin:$PATH
ENV PORT=5008
EXPOSE 5008
CMD ["python", "ad3.py"]
```

## Troubleshooting

### Docker Desktop not running (Windows)

**Error:** `The system cannot find the file specified` or `dockerDesktopLinuxEngine`

**Solution:**
1. Start Docker Desktop from the Start menu
2. Wait for Docker Desktop to fully start (whale icon in system tray should be steady)
3. Verify Docker is running:
   ```bash
   docker ps
   ```
4. Then retry:
   ```bash
   docker-compose up -d
   ```

### Container won't start
```bash
# Check logs
docker logs agent

# Check if port is already in use (Windows PowerShell)
netstat -an | Select-String "5008"
```

### Permission issues
```bash
# Run with user permissions
docker run -d -p 5008:5008 --user $(id -u):$(id -g) case-closed-agent
```

### Network issues
```bash
# Test connectivity from host
curl http://localhost:5008/

# Test from inside container
docker exec agent curl http://localhost:5008/
```

## Production Deployment

For production, consider:

1. **Use a production WSGI server** (gunicorn):
```dockerfile
RUN pip install gunicorn
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5008", "ad3:app"]
```

2. **Add health checks:**
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:5008/"]
  interval: 30s
  timeout: 10s
  retries: 3
```

3. **Resource limits** (1 CPU limit for competition - already in docker-compose.yml)

**Note:** `ad3.py` is single-threaded (sequential minimax search), so 1 CPU is sufficient and matches competition requirements.

4. **Logging:**
```bash
docker run -d -p 5008:5008 -v $(pwd)/logs:/app/logs case-closed-agent
```

