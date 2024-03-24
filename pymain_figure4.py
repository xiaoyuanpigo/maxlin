import os
os.system(" cp -r maxlin.c 3dcertify/ERAN/ELINA/fppoly/pool_approx.c ")
os.system(" cd 3dcertify/ERAN/ELINA/ && make all ")
for i in [0.010,0.012,0.013,0.014,0.015]:
    command=f"python __main__.py \
        --netname /data/code/benchmark-maxlin/nets/cifar_conv_maxpool.onnx \
        --epsilon {i}\
        --domain refinepoly \
        --dataset mnist \
        --complete False \
        --timeout_final_lp  100\
        --timeout_final_milp  100\
        --timeout_lp 1 \
        --timeout_milp 1\
        --use_default_heuristic True \
        --use_milp True \
        --n_milp_refine 1 \
        --sparse_n 70 \
        --k 3 \
        --s -2 \
        --num_params 0 \
        --approx_k True \
        --max_milp_neurons  30\
        --normalized_region False |tee figure4_{i}.log"
    os.system(command)
