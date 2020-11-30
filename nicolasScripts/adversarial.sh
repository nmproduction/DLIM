#!/bin/bash
source /usr/itetnas04/data-scratch-01/dlim_03hs20/data/conda/etc/profile.d/conda.sh
conda activate machineLearning

python ./train.py --dataroot ./datasets/facades --model adversarial_losses --name adv_og --direction BtoA --gan_mode original
python ./test.py --dataroot ./datasets/facades --direction BtoA --model adversarial_losses --name adv_og

python ./train.py --dataroot ./datasets/facades --model adversarial_losses --name adv_hinge --direction BtoA --gan_mode hinge
python ./test.py --dataroot ./datasets/facades --direction BtoA --model adversarial_losses --name adv_hinge 

python ./train.py --dataroot ./datasets/facades --model adversarial_losses --name adv_w --direction BtoA --gan_mode w
python ./test.py --dataroot ./datasets/facades --direction BtoA --model adversarial_losses --name adv_w 
 
python ./train.py --dataroot ./datasets/facades --model adversarial_losses --name adv_ls --direction BtoA --gan_mode ls
python ./test.py --dataroot ./datasets/facades --direction BtoA --model adversarial_losses --name adv_ls

conda activate fidelitySupport

python ./nicolasScripts/fidelityAdversarial.py
