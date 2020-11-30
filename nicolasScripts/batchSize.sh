#!/bin/bash
source /usr/itetnas04/data-scratch-01/dlim_03hs20/data/conda/etc/profile.d/conda.sh
conda activate machineLearning

python ./train.py --dataroot ./datasets/facades --model pix2pix --name batch1 --direction BtoA --batch_size 1
python ./train.py --dataroot ./datasets/facades --model pix2pix --name batch4 --direction BtoA --batch_size 4
python ./train.py --dataroot ./datasets/facades --model pix2pix --name batch16 --direction BtoA --batch_size 16
python ./train.py --dataroot ./datasets/facades --model pix2pix --name batch64 --direction BtoA --batch_size 64
python ./train.py --dataroot ./datasets/facades --model pix2pix --name batch256 --direction BtoA --batch_size 256

python ./test.py --dataroot ./datasets/facades --direction BtoA --model pix2pix --name batch1
python ./test.py --dataroot ./datasets/facades --direction BtoA --model pix2pix --name batch4
python ./test.py --dataroot ./datasets/facades --direction BtoA --model pix2pix --name batch16
python ./test.py --dataroot ./datasets/facades --direction BtoA --model pix2pix --name batch64
python ./test.py --dataroot ./datasets/facades --direction BtoA --model pix2pix --name batch256
