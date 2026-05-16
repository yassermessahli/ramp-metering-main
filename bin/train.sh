#!/usr/bin/bash

function run () {

python3 train.py -algo DuelingDoubleDQNAgent -max_total_steps 2000000

}



run

