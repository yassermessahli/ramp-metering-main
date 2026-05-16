# rl_env/dqn_config.py

# Assuming SUMO_PARAMS is needed for the CONFIG string used in save/log directories
# This import means dqn_config.py is likely in rl_env/ and custom_env is rl_env/custom_env/
import torch.nn as nn
import torch.optim as optim

from .custom_env.utils import (
    SUMO_PARAMS,
)  # Or just from .custom_env import SUMO_PARAMS if __init__.py exports it

# from torch import no_grad, as_tensor # Only needed if you were calculating n_flatten for CNN

# CONFIG will be used for naming save/log directories, e.g., "1ramp_1x3"
CONFIG = SUMO_PARAMS.get("config", "1ramp_1x3")

# """CHANGE HYPER PARAMETERS HERE""" ###################################################################################
# Review these carefully, especially step-based ones.
# One agent step = 40 simulation seconds.
# Example: If an episode is 3600s (SUMO_PARAMS["steps"]), then max_episode_steps = 3600 / 40 = 90
MAX_SIMULATION_SECONDS_PER_EPISODE = SUMO_PARAMS.get("steps", 3600)
AGENT_CONTROL_CYCLE_SEC = 40.0  # From RLController
MAX_AGENT_STEPS_PER_EPISODE = int(MAX_SIMULATION_SECONDS_PER_EPISODE / AGENT_CONTROL_CYCLE_SEC)


HYPER_PARAMS = {
    "gpu": "0",  # GPU #
    "n_env": 1,  # Multi-processing environments (usually 1 for SUMO unless carefully managed)
    "lr": 1e-4,  # Learning rate
    "gamma": 0.99,  # Discount factor
    "eps_start": 1.0,  # Epsilon start
    "eps_min": 0.01,  # Epsilon min (0.01 might be too low initially for harder problems)
    "eps_dec": 2e6,  # Epsilon decay steps (e.g., for ~1 million sim seconds: 1M/40 = 25k agent steps)
    # Adjust based on total training time desired.
    "eps_dec_exp": True,  # Epsilon exponential decay
    "bs": 32,  # Batch size (was 32, 64 is also common)
    "min_mem": 100000,  # Replay memory buffer min size (e.g., 10k agent steps * 40s/step = 400k sim seconds worth)
    # This means ~111 episodes of 3600s to fill if min_mem = 10k agent steps.
    "max_mem": 1000000,  # Replay memory buffer max size (100k agent steps * 40s/step = 4M sim seconds)
    "target_update_freq": 30000,  # Target network update frequency (in agent steps, e.g., every 500*40 = 20k sim seconds)
    "target_soft_update": True,  # Target network soft update
    "target_soft_update_tau": 1e-3,  # Target network soft update tau rate
    "save_freq": 10000,  # Save frequency (e.g., every 10 episodes, in agent steps)
    "log_freq": 4500,  # Log frequency (e.g., twice per episode, in agent steps)
    "save_dir": "./save/" + CONFIG + "/",  # Save directory
    "log_dir": "./logs/train/" + CONFIG + "/",  # Log directory
    "load": True,  # Load model if exists
    "repeat": 0,  # Repeat action (not applicable here as 1 action = 40s cycle)
    "max_episode_steps": 1000,  # Max agent steps (40s cycles) per episode
    "max_total_steps": 21e5,  # Max total training agent steps if > 0, else inf training
    # e.g., 50000 for 2M sim seconds of training (50000 * 40s)
    "algo": "DuelingDoubleDQNAgent",  # DQNAgent
    # DoubleDQNAgent
    # DuelingDoubleDQNAgent (Good choice)
    # PerDuelingDoubleDQNAgent (Good choice, but PER adds complexity)
}
########################################################################################################################


# """CHANGE NETWORK CONFIG HERE""" #####################################################################################
def network_config(input_dim_space):  # input_dim_space is the gym.spaces.Box object
    # """CHANGE NETWORK HERE""" ########################################################################################
    # This function defines an MLP for the 8-feature state vector.

    # Get the number of features from the input_dim_space (e.g., Box(shape=(8,)))
    # RLController.observation_space_n should be 8
    # CustomEnvWrapper creates observation_space = spaces.Box(..., shape=(observation_space_n,), ...)
    # So, input_dim_space.shape[0] will be 8.
    num_input_features = input_dim_space.shape[0]

    # Define hidden layer dimensions for the MLP
    # These are examples, tuning these is part of hyperparameter optimization
    fc_dims = (256, 128)  # Two hidden layers with 256 and 128 units respectively

    # Activation function
    activation = nn.ReLU()  # ReLU is common, ELU or LeakyReLU are alternatives

    # Construct the network
    net = nn.Sequential(
        nn.Linear(num_input_features, fc_dims[0]),
        activation,
        nn.Linear(fc_dims[0], fc_dims[1]),
        activation,
        # Add more layers if needed for more complex state-action mappings
        # nn.Linear(fc_dims[1], fc_dims[2]),
        # activation,
    )
    ####################################################################################################################

    # """CHANGE FC DUELING LAYER OUTPUT DIM HERE""" ####################################################################
    # This is the number of output features from the common body of the network,
    # before it splits into Value and Advantage streams (for DuelingDQN)
    # or before the final Q-value layer (for standard DQN).
    fc_out_dim = fc_dims[-1]  # Output dimension of the last hidden layer (e.g., 128)
    ####################################################################################################################

    # """CHANGE OPTIMIZER HERE""" ######################################################################################
    optim_func = optim.Adam  # Adam is a good default
    # optim_func = optim.RMSprop
    ####################################################################################################################

    # """CHANGE LOSS HERE""" ###########################################################################################
    loss_func = nn.SmoothL1Loss  # Huber loss, robust to outliers, common for DQN
    # loss_func = nn.MSELoss
    ####################################################################################################################

    return net, fc_out_dim, optim_func, loss_func


########################################################################################################################
