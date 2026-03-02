# Docker Setup for Highway Ramp Metering RL Project

## Prerequisites

1. Install Docker Desktop
2. Install NVIDIA Container Toolkit (for GPU support)
3. Make sure you have NVIDIA drivers installed

## Quick Start

### Building and Running with Docker Compose

1. Build the containers:
```bash
docker-compose build
```

2. Train the models:
```bash
docker-compose up rl-project
```

3. Run a trained model:
```bash
docker-compose up run-model
```

### Custom Commands

Run Q-Learning model:
```bash
docker-compose run --rm run-model python3 run_trained_model.py --model qlearning
```

Run DQN model:
```bash
docker-compose run --rm run-model python3 run_trained_model.py --model dqn
```

## Container Structure

The project is containerized with two services:

1. `rl-project`: For training models
   - CUDA-enabled PyTorch
   - SUMO traffic simulator
   - All Python dependencies

2. `run-model`: For running trained models
   - Same environment as rl-project
   - Configured for model inference

## Volume Mounts

- Project files: `.:/app`
- Model storage: `./models:/app/models`

## GPU Support

The containers are configured to use NVIDIA GPUs if available. Make sure you have:
1. NVIDIA GPU drivers installed
2. NVIDIA Container Toolkit installed
3. Docker configured to use NVIDIA runtime

## Troubleshooting

1. If SUMO GUI doesn't work:
   ```bash
   xhost +local:docker
   ```

2. If GPU is not detected:
   ```bash
   nvidia-smi
   ```
   Should show your GPU. If not, check NVIDIA drivers.

3. Memory issues:
   - Adjust Docker Desktop resources
   - Reduce batch size in training

## Development Workflow

1. Make changes to code locally
2. Container automatically syncs via volume mount
3. Changes take effect immediately
4. Models are saved to ./models directory

## Production Deployment

For production:
1. Remove volume mounts
2. Build final image with code included
3. Use specific version tags
4. Consider using Docker Swarm or Kubernetes

## Environment Variables

Can be set in docker-compose.yml:
- NVIDIA_VISIBLE_DEVICES
- SUMO_HOME
- PYTHONPATH

## Resource Management

GPU memory and CPU cores can be limited in docker-compose.yml:
```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
```
