# Optimal Control Course Project

This project implements various autonomous driving agents using the CARLA simulator, focusing on optimal control techniques for vehicle navigation.

## Installation Guide

### Step 1: Install CARLA Simulator

1. Download CARLA from the [official GitHub releases](https://github.com/carla-simulator/carla/releases). Choose version 0.9.13 or newer.

2. Extract the downloaded package to your preferred location:
   ```bash
   mkdir -p ~/CARLA
   tar -xvzf CARLA_0.9.13.tar.gz -C ~/CARLA
   ```

### Step 2: Create and Configure Conda Environment

```bash
# Create a new conda environment with Python 3.8
conda create -n carla-env python=3.8
conda activate carla-env

# Install required packages
conda install -c conda-forge numpy shapely matplotlib
conda install -c anaconda networkx
pip install pygame

# Additional packages that might be needed
pip install opencv-python
```

### Step 3: Install CARLA Python API

The CARLA Python API needs to be added to your Python path. There are two options:

#### Option 1: Install the CARLA Python package (Recommended)
```bash
# Add egg file to path - adjust path according to your CARLA installation
pip install ~/CARLA/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg
```

#### Option 2: Add to PYTHONPATH temporarily
```bash
# Add this to your .bashrc or run before executing scripts
echo 'export PYTHONPATH=$PYTHONPATH:/home/abdelrahman/CARLA_0.9.15/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg' >> ~/.bashrc
echo 'export PYTHONPATH=$PYTHONPATH:~/CARLA/PythonAPI/carla' >> ~/.bashrc
source ~/.bashrc
```

### Step 4: Project Setup
```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd oc_project

# Make sure you're using the carla-env
conda activate carla-env
```

## Running the Simulation

1. First, start the CARLA server:
```bash
cd ~/CARLA
./CarlaUE4.sh -quality-level=Low
```

2. Then, in a new terminal, run your script:
```bash
conda activate carla-env
cd ~/projects/oc_project
python carla_setup.py
```

## Troubleshooting

1. **ModuleNotFoundError: No module named 'carla'**
   - Make sure the CARLA Python API is correctly added to your Python path.

2. **Connection issues with CARLA server**
   - Ensure the CARLA server is running before executing your script.
   - Check if the port number (default: 2000) is correct in your code.

## Additional Resources

- [CARLA Documentation](https://carla.readthedocs.io/)
- [CARLA Python API Reference](https://carla.readthedocs.io/en/latest/python_api/)