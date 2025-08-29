#!/bin/bash

# This script runs a hyperparameter sweep for the Ares Prime project.
# It loops through different learning rates and gives each run a unique name.

echo "--- Starting Hyperparameter Sweep ---"

# Define the learning rates we want to test
LEARNING_RATES="0.001 0.0005 0.0001"

for lr in $LEARNING_RATES
do
    # Define a unique name for this run based on its parameters
    RUN_NAME="agent_lr_${lr}"

    echo ""
    echo "--- LAUNCHING RUN: ${RUN_NAME} ---"
    
    # Execute the main python script with the current parameters
    python main_executor.py --learning_rate $lr --run_name $RUN_NAME
done

echo ""
echo "--- Hyperparameter Sweep Complete ---"./run_sweep.sh