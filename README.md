# MFEBNet

## 0. Main Environments
```bash
conda create -n env_name python=3.8
conda activate env_name
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```

## 1. Prepare the dataset

- After downloading the datasets, you are supposed to put them into './dataset/', and the file format reference is as follows. (take the DeepGlobe dataset as an example.)

- '/dataset/'
  - train
    - images
      - ...
    - masks
      - ...
  - val
    - images
      - ...
    - masks
      - ...
  - test
    - images
      - ...
    - masks
      - ...

### CHN6_CUG：
- 512 × 512，The dataset is divided into a training set with 3608 images and a test/
validation set with 903 images.

### DeepGlobe：
- 1024x1024，There are a total of 6226 images, of which 4980 images are used for training, 623 images are used for testing, and 623 images are used for validation.
            

## 2. Train the MFEBNet
```python
python train.py
```

## 3. Obtain the outputs
- After trianing, you could obtain the results in './results/'
