#!/bin/bash
conda activate 3dcertify
conda info --env
echo "check your conda env if 3dcertify"
cp -f /root/MaxLin/maxlin.c /root/MaxLin/3dcertify/ERAN/ELINA/fppoly/pool_approx.c
cd /root/MaxLin/3dcertify/ERAN/ELINA
make all
cd /root/MaxLin/3dcertify
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!MaxLin!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

python verify_pointnet.py \
    --model models/pretrained-models/16p_natural.pth \
    --num_points 16 \
    --eps 0.005 \
    --pooling improved_max \
    --experiment example2
echo "******************************************************************************"
python verify_pointnet.py \
    --model models/pretrained-models/32p_natural.pth \
    --num_points 32 \
    --eps 0.005 \
    --pooling improved_max \
    --experiment example2
    echo "******************************************************************************"
python verify_pointnet.py \
    --model models/pretrained-models/64p_natural.pth \
    --num_points 64 \
    --eps 0.005 \
    --pooling improved_max \
    --experiment example2
   echo "******************************************************************************"
python verify_pointnet.py \
    --model models/pretrained-models/128p_natural.pth \
    --num_points 128 \
    --eps 0.005 \
    --pooling improved_max \
    --experiment example2
    echo "******************************************************************************"
python verify_pointnet.py \
    --model models/pretrained-models/256p_natural.pth \
    --num_points 256 \
    --eps 0.005 \
    --pooling improved_max \
    --experiment example2
    echo "******************************************************************************"
