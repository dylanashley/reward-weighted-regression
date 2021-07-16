#!/bin/bash

SCRIPT=$(dirname "$0")'/run.py'
TASKS_PER_FILE=20 #200

# assert command line arguments valid
if [ "$#" -gt "0" ]
    then
        echo 'usage: ./build.sh'
        exit
    fi

# begin amalgamating all tasks
TASKS_PREFIX='tasks_'
rm "$TASKS_PREFIX"*.sh 2>/dev/null
rm tasks.sh 2>/dev/null
I=0

###############################################################################

NUM_ITER='40'

for SEED in `seq 0 9`; do #99
  MODE='policy_iteration'
  USE_WANDB="$MODE"'_'"$SEED"
  RECORD_OUTFILE="$I"'.json'
  I=$((I + 1))
  ARGS=($MODE
        "--seed=$SEED"
        "--num-iter=$NUM_ITER"
        # "--use-wandb=$USE_WANDB"
        "--record-outfile=$RECORD_OUTFILE")
  echo 'python -O '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh

  MODE='reward_weighted_regression'
  USE_WANDB="$MODE"'_'"$SEED"
  RECORD_OUTFILE="$I"'.json'
  I=$((I + 1))
  ARGS=($MODE
        "--seed=$SEED"
        "--num-iter=$NUM_ITER"
        # "--use-wandb=$USE_WANDB"
        "--record-outfile=$RECORD_OUTFILE")
  echo 'python -O '"$SCRIPT"' '"${ARGS[*]}" >> tasks.sh
done

###############################################################################

# split tasks into files
perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' < tasks.sh > temp.sh
rm tasks.sh 2>/dev/null
split -l $TASKS_PER_FILE -a 3 temp.sh
rm temp.sh
AL=({a..z})
for i in `seq 0 25`; do
    for j in `seq 0 25`; do
        for k in `seq 0 25`; do
        FILE='x'"${AL[i]}${AL[j]}${AL[k]}"
        if [ -f $FILE ]; then
            ID=$((i * 26 * 26 + j * 26 + k))
            ID=${ID##+(0)}
            mv 'x'"${AL[i]}${AL[j]}${AL[k]}" "$TASKS_PREFIX""$ID"'.sh' 2>/dev/null
            chmod +x "$TASKS_PREFIX""$ID"'.sh' 2>/dev/null
        else
            break 3
        fi
        done
    done
done
echo $ID
