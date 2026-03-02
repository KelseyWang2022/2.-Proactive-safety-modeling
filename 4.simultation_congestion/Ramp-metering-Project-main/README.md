# Highway Ramp Metering Reinforcement Learning Project

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.9+-red.svg)
![SUMO](https://img.shields.io/badge/SUMO-v1.8.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Overview

This project implements Q-learning and Deep Q-Network (DQN) algorithms for controlling highway ramp metering using the SUMO traffic simulator. The goal is to optimize traffic flow by intelligently controlling when vehicles can enter the highway from on-ramps.

## Features

- Implementation of two RL algorithms:
  - Q-Learning (tabular approach)
  - Deep Q-Network (DQN)
- Real-time traffic simulation using SUMO
- Performance visualization and analysis
- Docker support for easy deployment
- Comprehensive documentation

## Quick Start

### Using Docker 

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RL-Project.git
cd RL-Project
```

2. Build and run with Docker:
```bash
docker-compose build
docker-compose up rl-project  # For training
# or
docker-compose up run-model   # For running trained model
```

See [DOCKER_README.md](DOCKER_README.md) for detailed Docker instructions.

### Manual Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install SUMO following instructions at: https://sumo.dlr.de/docs/Installing.html

3. Train the models:
```bash
python code/training/train_qlearning.py
python code/training/train_dqn.py
```

4. Run trained model:
```bash
python code/deployment/evaluate.py
```

## Project Structure

```
RL-Project/
├── train_and_evaluate.py    # Main training script
├── run_trained_model.py     # Script to run trained models
├── requirements.txt         # Package dependencies
├── README.md                # Project documentation
├── REPORT.md                # Detailed project report
├── DOCKER_README.md         # Docker setup instructions
├── docker-compose.yml       # Docker Compose configuration
├── Dockerfile               # Docker image definition
├── code/
│   ├── training/
│   │   ├── qlearning.py     # Q-Learning implementation
│   │   └── dqn.py           # Deep Q-Network implementation
│   └── utils/
│       └── env.py           # SUMO environment wrapper
├── config/
│   ├── highway.net.xml      # Highway network definition
│   ├── highway.rou.xml      # Traffic route definition
│   └── highway.sumocfg      # SUMO configuration file
└── models/
    ├── dqn_model.pth        # Trained DQN model weights
    └── qlearning_table.pkl  # Q-Learning table
```

## Documentation

- [Project Report](REPORT.md) - Detailed project analysis
- [Docker Guide](DOCKER_README.md) - Docker setup and usage
- [LaTeX Report](report.tex) - LaTeX version of the report

## Results

The project demonstrates successful traffic optimization using both Q-Learning and DQN approaches. Key findings:

- Improved traffic flow by up to X%
- Reduced average waiting time by Y%
- Better handling of peak traffic conditions

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- SUMO Team for the traffic simulation environment
- PyTorch Team for the deep learning framework

