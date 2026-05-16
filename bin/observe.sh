#!/usr/bin/bash

function run () {

SAVE="1ramp_1x3"

python3 observe.py -d save/$SAVE/DuelingDoubleDQNAgent_lr0.0001_model.pack -max_e 1 -log True -log_s 1

}

# cd ..

# source venv/bin/activate

run

# deactivate

# exit
