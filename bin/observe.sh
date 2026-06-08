#!/usr/bin/bash

function run () {

MODEL_PATH="save/1ramp_1x3/lane_control_macro_only/DuelingDoubleDQNAgent_lr0.0001_model.pack"
uv run observe.py -d $MODEL_PATH -max_e 1 -log True -log_s 1
}

# cd ..

# source venv/bin/activate

run

# deactivate

# exit
