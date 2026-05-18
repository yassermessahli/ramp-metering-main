#!/usr/bin/bash

function run () {

uv run train.py -algo DuelingDoubleDQNAgent -load False -max_total_steps 1000 -min_mem 100 

}



run

