#!/bin/bash
source /usr/itetnas04/data-scratch-01/dlim_03hs20/data/conda/etc/profile.d/conda.sh
conda activate machineLearning

python ./train.py --dataroot ./datasets/facades --model pix2pix --name superdupermodel --direction BtoA

python ./test.py --dataroot ./datasets/facades --direction BtoA --model pix2pix --name superdupermodel 
python ./test.py --dataroot ./datasets/facades --direction BtoA --model pix2pix --name facades_label2photo_pretrained

python ./nicolasScripts/fidelity.py
