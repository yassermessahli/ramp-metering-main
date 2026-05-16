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
    "save_freq": 10000,  # Save frequency
    "log_freq": 4500,  # Log frequency
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


# --- NETWORK CONFIGURATION FUNCTION ---
# Place this custom class definition at the top of dqn_config.py,
# right after the imports.

import torch as T  # Make sure T is imported as torch


class TwoStreamHybridNetwork(nn.Module):
    def __init__(self, macro_vec_len, micro_shape_chw, cnn_params, dense_params, activation_fn):
        """
        Initializes the two-stream network.
        Args:
            macro_vec_len (int): The length of the macro state vector (e.g., 14).
            micro_shape_chw (tuple): The shape of the micro grid in (Channels, Height, Width) format (e.g., (2, 27, 5)).
            cnn_params (list of tuples): Parameters for the CNN layers.
            dense_params (list of tuples): Parameters for the final dense layers.
            activation_fn: The activation function to use (e.g., nn.ReLU()).
        """
        super().__init__()

        self.macro_len = macro_vec_len
        self.micro_shape = micro_shape_chw
        self.micro_flat_len = micro_shape_chw[0] * micro_shape_chw[1] * micro_shape_chw[2]

        # --- Define the Micro (CNN) Stream ---
        cnn_layers = []
        in_channels = self.micro_shape[0]  # Should be 2
        for filters, kernel, stride in cnn_params:
            padding = (kernel[0] // 2, kernel[1] // 2)
            cnn_layers.append(
                nn.Conv2d(in_channels, filters, kernel_size=kernel, stride=stride, padding=padding)
            )
            cnn_layers.append(activation_fn)
            in_channels = filters

        self.cnn_stream = nn.Sequential(*cnn_layers)

        # --- Calculate the size of the concatenated feature vector (CORRECTED) ---
        with T.no_grad():
            dummy_micro_input = T.zeros(1, *self.micro_shape)  # e.g., (1, 2, 27, 5)
            # --- FIX: Flatten the CNN output to get the feature vector size ---
            cnn_output_flat = self.cnn_stream(dummy_micro_input).flatten(start_dim=1)
            cnn_output_size = cnn_output_flat.shape[1]

        concatenated_size = cnn_output_size + self.macro_len

        # --- Define the Final (Dense) Stream ---
        dense_layers = []
        in_features = concatenated_size
        for out_features in dense_params:
            dense_layers.append(nn.Linear(in_features, out_features))
            dense_layers.append(activation_fn)
            in_features = out_features

        self.dense_stream = nn.Sequential(*dense_layers)

        self.fc_out_dim = dense_params[-1] if dense_params else in_features

    # --- THIS IS THE ONLY FORWARD FUNCTION YOU SHOULD HAVE ---
    def forward(self, x):
        """
        The forward pass that unpacks the flat vector and processes it.
        Args:
            x (Tensor): The flattened state vector of shape (batch_size, 284).
        """
        # --- 1. Unpack the State ---
        macro_input = x[:, : self.macro_len]
        micro_flat_input = x[:, self.macro_len :]

        micro_unpacked = micro_flat_input.view(-1, *self.micro_shape)

        # --- 2. Process the Micro Stream ---
        processed_micro_4d = self.cnn_stream(micro_unpacked)

        # --- FIX: Flatten the 4D CNN output to a 2D tensor before concatenation ---
        processed_micro_2d = processed_micro_4d.flatten(start_dim=1)

        # --- 3. Concatenate ---
        # Now we are concatenating a 2D tensor with another 2D tensor
        combined_features = T.cat([processed_micro_2d, macro_input], dim=1)

        # --- 4. Final Processing ---
        output = self.dense_stream(combined_features)

        return output


# In dqn_config.py, replace the ENTIRE network_config function with this.


def network_config(input_dim):
    # input_dim is the shape of the observation space from the gym wrapper,
    # which will be a tuple like (284,) for the flat vector.

    # """DEFINE YOUR ARCHITECTURE PARAMETERS HERE""" ##################################################################

    # --- Part 1: Define the dimensions ---
    # These must match the state creation in rl_controller.py
    MACRO_VECTOR_LENGTH = 14
    # IMPORTANT: PyTorch CNNs expect Channels-First: (Channels, Height, Width)
    MICRO_GRID_SHAPE_CHW = (
        SUMO_PARAMS["grid_channels"],
        SUMO_PARAMS["grid_rows"],
        SUMO_PARAMS["grid_cols"],
    )  # (2, 27, 5)

    # --- Part 2: Define the network layers ---

    # Parameters for the CNN stream: (filters, kernel_size, stride)
    # Using the professionally recommended architecture
    CNN_PARAMS = [
        (32, (3, 3), (1, 1)),
        (64, (3, 3), (2, 1)),  # Asymmetric stride
        (64, (3, 3), (2, 2)),
    ]

    # Parameters for the final dense layers (after concatenation)
    DENSE_PARAMS = [
        512,
        256,
    ]  # A single hidden layer of 512 units. Add more if needed, e.g., [512, 256]

    # --- Part 3: Choose Activation, Optimizer, Loss ---

    ACTIVATION = nn.ELU()  # ReLU is standard, but ELU is fine too.
    OPTIMIZER = optim.Adam
    LOSS_FUNCTION = nn.SmoothL1Loss

    ####################################################################################################################

    # --- Instantiate the custom network ---
    net = TwoStreamHybridNetwork(
        macro_vec_len=MACRO_VECTOR_LENGTH,
        micro_shape_chw=MICRO_GRID_SHAPE_CHW,
        cnn_params=CNN_PARAMS,
        dense_params=DENSE_PARAMS,
        activation_fn=ACTIVATION,
    )

    # The fc_out_dim is now an attribute of our custom network
    fc_out_dim = net.fc_out_dim

    return net, fc_out_dim, OPTIMIZER, LOSS_FUNCTION


########################################################################################################################
