#!/usr/bin/bash

function run () {

uv run train.py -algo DuelingDoubleDQNAgent -load False -max_total_steps 100000 -min_mem 10000 

# to use an existing buffer
# uv run python train.py -load_buffer True
}



run

