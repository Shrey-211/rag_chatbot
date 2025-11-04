# Docker Build Optimization Guide

This project uses several optimizations to speed up Docker builds:

## Key Optimizations

### 1. **BuildKit Cache Mounts**
- **APT packages**: Uses cache mounts for `/var/cache/apt` and `/var/lib/apt`
- **Python packages**: Uses cache mount for `/root/.cache/pip`
- **Node packages**: Uses cache mount for `/root/.npm` and `/app/node_modules/.vite`

### 2. **Layer Caching Strategy**
- Dependencies are installed before copying application code
- This ensures dependency layers are cached and only rebuilt when dependencies change
- Application code is copied last (most frequently changing layer)

### 3. **Optimized .dockerignore Files**
- Excludes unnecessary files from build context
- Reduces build context size significantly
- Prevents copying large files like `node_modules/`, `data/`, etc.

### 4. **Multi-stage Builds**
- Separates development and production stages
- Allows reusing base layers across stages

## Enabling BuildKit

To get the full benefit of cache mounts, ensure BuildKit is enabled:

### Linux/Mac
```bash
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
```

Or add to your `~/.bashrc` or `~/.zshrc`:
```bash
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
```

### Windows (PowerShell)
```powershell
$env:DOCKER_BUILDKIT=1
$env:COMPOSE_DOCKER_CLI_BUILD=1
```

Or set permanently:
```powershell
[Environment]::SetEnvironmentVariable("DOCKER_BUILDKIT", "1", "User")
[Environment]::SetEnvironmentVariable("COMPOSE_DOCKER_CLI_BUILD", "1", "User")
```

### Docker Desktop
BuildKit is enabled by default in Docker Desktop 20.10+

## Build Commands

### First Build (slower, downloads everything)
```bash
docker-compose build
```

### Subsequent Builds (much faster with cache)
```bash
docker-compose build
```

### Force Rebuild (ignore cache)
```bash
docker-compose build --no-cache
```

### Build Specific Service
```bash
docker-compose build rag-api
docker-compose build webapp
```

## Performance Tips

1. **Only rebuild when needed**: Code changes don't require rebuilding if using volume mounts in development
2. **Use specific tags**: Pin base image versions (e.g., `python:3.11-slim` instead of `python:3-slim`)
3. **Separate dev/prod requirements**: If you have many dev dependencies, consider splitting `requirements.txt`
4. **Parallel builds**: Docker Compose builds services in parallel when possible

## Expected Build Times

- **First build**: ~5-10 minutes (downloads all dependencies)
- **Subsequent builds** (with cache): ~30 seconds - 2 minutes (only rebuilds changed layers)
- **Code-only changes**: ~10-30 seconds (only application code layer rebuilds)

## Troubleshooting

If builds are still slow:

1. **Check BuildKit is enabled**:
   ```bash
   docker buildx version
   ```

2. **Clear cache if needed**:
   ```bash
   docker builder prune
   ```

3. **Check .dockerignore**: Ensure large files aren't being copied
   ```bash
   docker build --progress=plain . 2>&1 | grep "Sending build context"
   ```

4. **Inspect cache usage**:
   ```bash
   docker system df
   ```

