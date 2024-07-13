
# Towards General Robustness Verification of MaxPool-based Convolutional Neural Networks via Tightening Linear Approximation (CVPR 2024)
## Setup Instructions

Clone this repository, including all submodules:
```bash
mkdir MaxLin
cd MaxLin
git clone https://github.com/xiaoyuanpigo/MaxLin.git
```

To test MaxLin on CNN-Cert framework [1]:
```bash
git clone https://github.com/AkhilanB/CNN-Cert.git
cd ..
cp -f cnncert_MaxLin.py CNN-Cert/cnn_bounds_full_core.py
```

To test MaxLin on ERAN framework (https://github.com/eth-sri/eran):
```bash
git clone --recurse-submodules https://github.com/eth-sri/3dcertify.git
cd ..
cp -f verify_perturbation.py 3dcertify/verify_perturbation.py 
cp -f elina_maxlin.c 3dcertify/ERAN/ELINA/fppoly/pool_approx.c
cd 3dcertify/ERAN/ELINA/
make all
cd ../../..
```

To test MaxLin on $\alpha,\beta$-CROWN framework (https://github.com/Verified-Intelligence/alpha-beta-CROWN):
```bash
git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
cd ..
cp -f auto_lirpa_maxlin.py alpha-beta-CROWN/complete_verifier/auto_LiRPA/operators/pooling.py
```

Create a [conda](https://www.anaconda.com/products/individual) environment with the required dependencies:
```bash
conda env create -f cnncert-env.yml
conda activate cnncert
conda env create -f 3dcertify-env.yml
conda activate 3dcertify
```

Download the CNN models used in the paper. We use CNN-Cert(https://www.dropbox.com/s/mhe8u2vpugxz5ed/models.zip), 3DCertify(https://files.sri.inf.ethz.ch/pointclouds/pretrained-models.zip), and ERAN (accessed in https://github.com/eth-sri/eran) benchmarks.

Some models(.h5) need to be transformed into other typt(.pb)
```bash
cd ../CNN-Cert/models
python h52pb.py
```

## Run Certification

### The results in Table 1

The results of MaxLin
```bash
cp -f maxlin.py CNN-Cert/cnn_bounds_full_core.py 
cd CNN-Cert
python pymain_new_network.py
```

### The results in Table 2
```bash
nohup ./table2.sh > ./table2result.log &
```

### The results in Figure 4
```bash
python pymain_figure4.py
```
 
### The results in Figure 5(after generating properties)
```bash
python pymain_figure5.py
```

### The results in Table 4
```bash
nohup ./table4.sh > ./table4result.log &
```
[1] Akhilan Boopathy, Tsui-Wei Weng, Pin-Yu Chen, Sijia Liu, and Luca Daniel. Cnn-cert: an efficient framework for certifying robustness of convolutional neural networks. In Proceedings of the AAAI Conference on Artificial Intelligenc (AAAI), pages 3240–3247, 2019.



## Citation
If you find our work helpful, please consider citing 

```bash
@inproceedings{xiao2024towards,
  title={Towards General Robustness Verification of MaxPool-based Convolutional Neural Networks via Tightening Linear Approximation},
  author={Xiao, Yuan and Ma, Shiqing and Zhai, Juan and Fang, Chunrong and Jia, Jinyuan and Chen, Zhenyu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={24766--24775},
  year={2024}
}
```
