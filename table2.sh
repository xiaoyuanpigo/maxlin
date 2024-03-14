
#!/bin/bash

conda info --env
echo "check your conda env if 3dcertify"
    
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MaxLin~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

echo "MaxLin"
echo "------------------------------------------------------------------------------"
echo "------------------------------------------------------------------------------"
cp -f /root/MaxLin/maxlin.c /root/MaxLin/3dcertify/ERAN/ELINA/fppoly/pool_approx.c
cd /root/3dcertify/ERAN/ELINA
make all
cd /root/MaxLin/3dcertify


echo "mnist conv_maxpool"
echo "******************************************************************************"
echo "******************************************************************************"
python verify_perturbation.py \
    --model /root/3dcertify/models/mnist_conv_maxpool.onnx\
    --num_points 64 \
    --eps 0.01 \
    --pooling max \
    --experiment example2
    
echo "mnist layer=4"
echo "******************************************************************************"
echo "******************************************************************************"
cd /root/3dcertify
python verify_perturbation.py \
    --model /root/3dcertify/models/mnist_cnn_4layer.pb\
    --num_points 64 \
    --eps 0.01 \
    --pooling max \
    --experiment example2

echo "mnist layer=5"
echo "******************************************************************************"
echo "******************************************************************************"
python verify_perturbation.py \
    --model /root/3dcertify/models/mnist_cnn_5layer.pb\
    --num_points 64 \
    --eps 0.01 \
    --pooling max \
    --experiment example2
    
echo "mnist layer=6"
echo "******************************************************************************"
echo "******************************************************************************"
python verify_perturbation.py \
    --model /root/3dcertify/models/mnist_cnn_6layer.pb\
    --num_points 64 \
    --eps 0.01 \
    --pooling max \
    --experiment example2
    
echo "cifar layer=4"
echo "******************************************************************************"
echo "******************************************************************************"
python verify_perturbation.py \
    --model /root/3dcertify/models/cifar_conv_maxpool.onnx\
    --num_points 64 \
    --eps 0.01 \
    --pooling max \
    --experiment example2
    
echo "cifar layer=4"
echo "******************************************************************************"
echo "******************************************************************************"
python verify_perturbation.py \
    --model /root/3dcertify/models/cifar_cnn_4layer.pb\
    --num_points 64 \
    --eps 0.01 \
    --pooling max \
    --experiment example2

echo "cifar layer=5"
echo "******************************************************************************"
echo "******************************************************************************"
python verify_perturbation.py \
    --model /root/3dcertify/models/cifar_cnn_5layer.pb\
    --num_points 64 \
    --eps 0.01 \
    --pooling max \
    --experiment example2

echo "cifar layer=6"
echo "******************************************************************************"
echo "******************************************************************************"
cd /root/3dcertify
python verify_perturbation.py \
    --model /root/3dcertify/models/cifar_cnn_6layer.pb\
    --num_points 64 \
    --eps 0.01 \
    --pooling max \
    --experiment example2

