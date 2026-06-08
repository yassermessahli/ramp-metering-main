#!/usr/bin/bash

function run () {

MODEL_PATH="save/1ramp_1x3/lane_control_macro_only/DuelingDoubleDQNAgent_lr0.0001_model.pack"
uv run python diagnostics/observe_smooth.py -d $MODEL_PATH -seed 42

}

# cd ..

# source venv/bin/activate

run

# deactivate

# exit
