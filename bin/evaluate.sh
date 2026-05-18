
function run () {

ALGO1="AlwaysGreenBaseline"
ALGO2="FixedCycleBaseline"
ALGO3="AlineaDsBaseline"
ALGO4="PiAlineaDsBaseline"
ALGO5="DQNAgent"
MODELPATH="save/1ramp_1x3/DuelingDoubleDQNAgent_lr0.0001_model.pack"
EPISODES=50

uv run evaluate.py -s $ALGO5 -d $MODELPATH -n $EPISODES 


}

run