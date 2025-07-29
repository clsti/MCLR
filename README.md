# Modeling and Control of Legged Robots - Quadruped Walking Control Project

This project implements walking control for the Go2 quadruped robot using optimal control methods. It supports both Open-loop Optimal Control (OPC) and Model Predictive Control (MPC) approaches with trajectory planning and visualization capabilities.

## Environment Setup

Due to conflicts with ROS, this project uses a dedicated Conda environment. Install the required dependencies:

```bash
# Create and activate conda environment
conda create -n walking python=3.8
conda activate walking

# Install core dependencies
conda install -c conda-forge pinocchio
conda install -c conda-forge pybullet
conda install -c conda-forge crocoddyl
```

Additional dependencies may be required depending on your system configuration.

## Configuration

### Control Parameters
The `conf_go2.py` file contains all configurable parameters:

- **Control Mode**: Switch between OPC and MPC
  ```python
  mpc_enabled = True  # True for MPC, False for OPC
  ```

- **Walking Parameters**:
  - `step_length`: Forward step distance (m)
  - `step_height`: Maximum foot lift height (m)
  - `n_per_step`: Number of time steps per leg movement
  - `n_steps`: Number of steps to plan (OPC only)

- **MPC Parameters**:
  - `mpc_horizon`: Planning horizon (number of full gait cycles)
  - `mpc_replan_frequency`: How often to replan (in full gait cycles)
  - `mpc_warm_start`: Enable warm starting with previous solution

- **Simulation Frequencies**:
  - `sim_freq`: Simulation frequency (Hz)
  - `control_freq`: Control frequency (Hz)

## Usage

### Running the Simulation
Execute the main script:
```bash
python main.py
```

### Trajectory Visualization
You can enable/disable trajectory visualization in `main.py`:

```python
# In run_mpc_control function
VISU = True  # Set to False to disable visualization

# In run_opc_control function  
VISU = True  # Set to False to disable visualization
```

**Warning**: Visualization may slow down the simulation, especially with large `n_per_step` values.

## Code Structure

### Core Files

#### `main.py`
Main execution script that:
- Initializes the PyBullet simulator and Go2 robot
- Creates the controller and trajectory planner
- Runs either MPC or OPC control loop based on configuration
- Handles trajectory visualization

#### `conf_go2.py`
Configuration file containing:
- Robot parameters and URDF paths
- PD controller gains
- Simulation and control frequencies
- Walking gait parameters
- MPC/OPC settings

#### `go2.py`
Go2 robot interface that:
- Initializes the robot model using example-robot-data
- Manages PyBullet simulation interface
- Handles state estimation (position, velocity, COM)
- Applies torque commands with PD control

#### `controller.py` (Go2Controller)
Optimal control implementation featuring:
- Crocoddyl-based action models for walking
- Contact and swing phase dynamics
- Cost functions for tracking and regularization
- Solver interface for optimal control problems
- Support for both forward and inverse dynamics

#### `foot_trajectory_planner.py` (TrajectoriesPlanner)
Trajectory planning module that:
- Generates foot swing trajectories with specified step length/height
- Plans center-of-mass (COM) motion
- Supports multi-step planning for both OPC and MPC
- Handles gait cycle coordination (RR → LF → RF → LH sequence)
- Provides trajectory visualization capabilities

#### `mpc_controller.py` (MPCController)
Model Predictive Control implementation that:
- Recalculates optimal control problems at specified intervals
- Manages warm starting with previous solutions
- Tracks gait cycles and leg coordination
- Provides reactive control compared to pre-planned OPC
- Handles replanning triggers and control execution

## Control Approaches

### Open-loop Optimal Control (OPC)
- Pre-plans entire walking sequence offline
- Solves one large optimal control problem
- Executes planned trajectory open-loop
- Faster execution but less adaptive to disturbances

### Model Predictive Control (MPC)
- Replans trajectory online at specified intervals
- Uses receding horizon optimization
- More reactive to state changes and disturbances
- Higher computational cost but improved robustness

## Gait Pattern

The implementation uses a trotting gait with the following leg sequence:
1. **RR** (Rear Right) - swing phase
2. **LF** (Left Front) - swing phase  
3. **RF** (Right Front) - swing phase
4. **LH** (Left Hind) - swing phase

Each leg movement consists of `n_per_step` time steps plus one foot switch action.

## Key Features

- **Warm Starting**: MPC can use previous solutions as initial guesses to improve convergence
- **Trajectory Visualization**: Real-time display of planned foot and COM trajectories
- **Configurable Gait**: Adjustable step length, height, and timing parameters
- **Robust Control**: PD control with feedforward torques from optimal control
- **Multi-Phase Planning**: Handles both swing and stance phases with proper contact constraints

## Troubleshooting

### Common Issues

1. **Path Errors**: Ensure all paths in `conf_go2.py` point to correct locations
2. **Visualization Slowdown**: Disable visualization (`VISU = False`) for faster simulation
3. **MPC Convergence**: Try adjusting horizon length or enabling warm start
4. **Control Instability**: Check PD gains and time step parameters

### Performance Tips

- Use smaller horizons for faster MPC replanning
- Reduce `n_per_step` for quicker gait cycles
- Disable visualization for maximum simulation speed
- Enable warm starting for better MPC convergence

## Dependencies

- **Pinocchio**: Rigid body dynamics and kinematics
- **PyBullet**: Physics simulation
- **Crocoddyl**: Optimal control framework
- **NumPy**: Numerical computations
- **Example Robot Data**: Robot model definitions

