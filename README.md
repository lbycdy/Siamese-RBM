# Siamese-RBM



## Installation

Please find installation instructions in [`INSTALL.md`](INSTALL.md).

## Quick Start:

```bash
export PYTHONPATH=/path/to/Siamese-RBM:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### Download models

Download models in [Model Zoo](MODEL_ZOO.md) and put the `model.pth` in the correct directory in experiments


```bash
cd experiments/siamese_r50_l234lb

```
###  Training :wrench:
```
`` bash
python ../../tools/lbtrainDogCatHorseAsPersonV017PerTKMobileOneWeightAddS16S32CATS8singleclass4gpu_ACMOutPointMaskCROPBBnoRKDropout_HeadPadding.py 	 \
	--dataset ../../data/Cows        \ # dataset name
 ```

### Test tracker
```
`` bash
python -u ../../tools/test.py 	\
	--snapshot ****.pth 	\ # model path
	--dataset ../../data/Cows  	\ # dataset name
	--config config.yaml	  # config file
```

The testing results will in the current directory(results/dataset/model_name/)


### Demo
```
assume still in experiments/siamban_r50_l234

`` bash
python ../../tools/lbdemoDogCatHorseAsPersonV017PerTKMobileOneWeightAddsingleclass4gpu_ACMOutPointMask_HeadPadding1.py	 \
	--dataset ../../data/Cows         \ # dataset name
