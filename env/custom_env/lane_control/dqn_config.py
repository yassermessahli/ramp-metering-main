import torch as T
import torch.nn as nn
import torch.optim as optim

from ..utils import SUMO_PARAMS

CONFIG = SUMO_PARAMS.get("config", "1ramp_1x3")

# One agent step = 40 simulation seconds.
MAX_SIMULATION_SECONDS_PER_EPISODE = SUMO_PARAMS.get("steps", 3600)
AGENT_CONTROL_CYCLE_SEC = 40.0
MAX_AGENT_STEPS_PER_EPISODE = int(
    MAX_SIMULATION_SECONDS_PER_EPISODE / AGENT_CONTROL_CYCLE_SEC
)

# Variant tag is appended to save/log dirs so this experiment does not
# collide with the prior "micro + macro lane" variant's checkpoints/logs.
VARIANT_TAG = "lane_control"

HYPER_PARAMS = {
    "gpu": "0",
    "n_env": 1,
    "lr": 1e-4,
    "gamma": 0.99,
    "eps_start": 1.0,
    "eps_min": 0.01,
    "eps_dec": 2e6,
    "eps_dec_exp": True,
    "bs": 32,
    "min_mem": 100000,
    "max_mem": 1000000,
    "target_update_freq": 30000,
    "target_soft_update": True,
    "target_soft_update_tau": 1e-3,
    "save_freq": 10000,
    "log_freq": 4500,
    "save_dir": "./save/" + CONFIG + "/" + VARIANT_TAG + "/",
    "log_dir": "./logs/train/" + CONFIG + "/" + VARIANT_TAG + "/",
    "load": True,
    "repeat": 0,
    "max_episode_steps": 1000,
    "max_total_steps": 21e5,
    "algo": "DuelingDoubleDQNAgent",
}


class TwoStreamHybridNetwork(nn.Module):
    def __init__(
        self, macro_vec_len, micro_shape_chw, cnn_params, dense_params, activation_fn
    ):
        super().__init__()

        self.macro_len = macro_vec_len
        self.micro_shape = micro_shape_chw
        self.micro_flat_len = (
            micro_shape_chw[0] * micro_shape_chw[1] * micro_shape_chw[2]
        )

        cnn_layers = []
        in_channels = self.micro_shape[0]
        for filters, kernel, stride in cnn_params:
            padding = (kernel[0] // 2, kernel[1] // 2)
            cnn_layers.append(
                nn.Conv2d(
                    in_channels,
                    filters,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                )
            )
            cnn_layers.append(activation_fn)
            in_channels = filters

        self.cnn_stream = nn.Sequential(*cnn_layers)

        with T.no_grad():
            dummy_micro_input = T.zeros(1, *self.micro_shape)
            cnn_output_flat = self.cnn_stream(dummy_micro_input).flatten(start_dim=1)
            cnn_output_size = cnn_output_flat.shape[1]

        concatenated_size = cnn_output_size + self.macro_len

        dense_layers = []
        in_features = concatenated_size
        for out_features in dense_params:
            dense_layers.append(nn.Linear(in_features, out_features))
            dense_layers.append(activation_fn)
            in_features = out_features

        self.dense_stream = nn.Sequential(*dense_layers)
        self.fc_out_dim = dense_params[-1] if dense_params else in_features

    def forward(self, x):
        macro_input = x[:, : self.macro_len]
        micro_flat_input = x[:, self.macro_len :]
        micro_unpacked = micro_flat_input.view(-1, *self.micro_shape)

        processed_micro_4d = self.cnn_stream(micro_unpacked)
        processed_micro_2d = processed_micro_4d.flatten(start_dim=1)

        combined_features = T.cat([processed_micro_2d, macro_input], dim=1)
        output = self.dense_stream(combined_features)
        return output


def network_config(input_dim):
    # input_dim is (284,) — unchanged from the active variant.
    # Action head width is set elsewhere via env.action_space.n (= 16 here).
    MACRO_VECTOR_LENGTH = 14
    MICRO_GRID_SHAPE_CHW = (
        SUMO_PARAMS["grid_channels"],
        SUMO_PARAMS["grid_rows"],
        SUMO_PARAMS["grid_cols"],
    )

    CNN_PARAMS = [
        (32, (3, 3), (1, 1)),
        (64, (3, 3), (2, 1)),
        (64, (3, 3), (2, 2)),
    ]
    DENSE_PARAMS = [512, 256]

    ACTIVATION = nn.ELU()
    OPTIMIZER = optim.Adam
    LOSS_FUNCTION = nn.SmoothL1Loss

    net = TwoStreamHybridNetwork(
        macro_vec_len=MACRO_VECTOR_LENGTH,
        micro_shape_chw=MICRO_GRID_SHAPE_CHW,
        cnn_params=CNN_PARAMS,
        dense_params=DENSE_PARAMS,
        activation_fn=ACTIVATION,
    )

    fc_out_dim = net.fc_out_dim
    return net, fc_out_dim, OPTIMIZER, LOSS_FUNCTION
