#!/usr/bin/zsh
python test.py --dataset DISFA --arc resnet50 \
 --exp-name test_fold3_0_10 \
 --resume ~/Downloads/ME-GraphAU_resnet50_DISFA/MEFARG_resnet50_DISFA_fold1.pth \
 --fold 1
