#!/bin/bash
source /usr/itetnas04/data-scratch-01/dlim_03hs20/data/conda/etc/profile.d/conda.sh
conda activate machineLearning

python ./test.py --dataroot ./datasets/facades --direction BtoA --model stochastic --name stoch 
