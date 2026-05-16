"""
- change the current folder for the simulation of 1x3 and 2x3 (bring from \ramp)
- add the 2x2x3 simulation
- fix the demand strategy (fixed demand or generated demand)
- add or change the demand strategy to the config file
- tune the sumo params
    - steps
    - veh_per_hour
    - communication range
"""

CONFIGS_SIMPLE = [
    "1ramp_1x3",
    "1ramp_2x3" # Configuration for a simple network with 1 traffic light and a 4x4 grid layout.
]

CONFIGS_MULTI = [
    "3ramp_1x3",
    "3ramp_2x3"
]

SUMO_PARAMS = {
    "config": CONFIGS_SIMPLE[0], # The SUMO configuration to use.
    "log_overall_metrics" : True, # Whether to log overall metrics for the simulation.
    "steps": 3600, # The number of simulation steps to run.
    "delay": 0,   # The delay (in milliseconds) between simulation steps when running with a GUI.
    "gui": False,  # Whether to run SUMO with a graphical user interface (GUI).
    "log": False, # Whether to enable logging of simulation statistics.
    "seed":False, # Whether to use a fixed seed for the simulation (True for fixed, False for random).
    "seed_value": 42, # The seed value to use for the simulation if `seed` is True.
    "alinea_detector_period_sec": 40.0,


    # Base values for flows
    "veh_per_hour_main": [4000, 4500, 5000, 5500, 6000, 6500],
    "veh_per_hour_on_ramp": [1400, 1500, 1600, 1700, 1800, 1900, 2000],
    "veh_per_hour_off_ramp": [100, 300, 500],

    "generate_route_file": True,


    # Biased towards higher flows as requested.
    "veh_per_hour_main_weights": [0.05, 0.1, 0.15, 0.3, 0.25, 0.15], # Sums to 1.0
    "veh_per_hour_on_ramp_weights": [0.05, 0.05, 0.05, 0.15, 0.25, 0.3, 0.15], # Sums to 1.0
    "veh_per_hour_off_ramp_weights": [0.4, 0.4, 0.2], # Sums to 1.0 (Example weights)


    "con_penetration_rate_range": [0.01, 0.99],  ## updated

    "v_type_def": "def", # The ID of the default vehicle type.
    "v_type_con": "con", # The ID of the connected vehicle type.

    # Physical Characteristics of the Vehicle
    "v_length": 5,     # The length of the vehicles (in meters).
    "v_min_gap": 2.5,   # The minimum gap (in meters) between vehicles.
    "v_max_speed": 35, # The maximum speed of the vehicles (in m/s, ~100 km/h).
    "rnd":(False, False),

    "con_range": 216.0, # Communication range for Connected vehicles (in meters).
    "cell_length": 8, # Length of a cell on the grid (in meters).
    "grid_cols" : 5,
    "grid_channels" : 2,
    "grid_rows" : 27, # Number of rows in the grid. (int(216/8))
    "vector_len" :14,

    "observation_shape_macro": (14,),
    "observation_shape_micro": (2, 27, 5) # Channels-first for PyTorch
}
