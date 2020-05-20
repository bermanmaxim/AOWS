# AOWS
**AOWS: Adaptive and optimal network width search with latency constraints**, Maxim Berman, Leonid Pishchulin, Ning Xu, Matthew B. Blaschko, Gérard Medioni, _NAS workshop @ ICLR 2020_ and _CVPR 2020 (oral)_.


<img src="https://user-images.githubusercontent.com/5989894/82336026-5a9f2e80-99ea-11ea-8141-facbcf9fd60d.gif" width="350" alt="AOWS-teaser">

## Usage

### Latency model
_main file: `latency.py`_

_depends on: PyTorch, [CVXPY](https://www.cvxpy.org/), matplotlib, numpy, [numba](http://numba.pydata.org/), scipy, [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt/tree/e22844a449a880123435fce7e6444f1516ebbe60)_


* Generate training and validation samples
```Shell
python latency.py generate --device trt --dtype fp16 \
   --biased --count 8000 output/samples_trt16.jsonl
python latency.py generate --device trt --dtype fp16 \
   --count 200 output/val_trt16.jsonl
```
* Fit the model (`K` controls the amount of regularization and should be set by validation)
```Shell
python latency.py fit output/samples_trt16.jsonl \
   output/model_trt16_K100.0.jsonl -K 100.0
```
* Validate the model (produces a plot)
```Shell
python latency.py validate output/val_trt16.jsonl \
   output/model_trt16_K100.0.jsonl output/correlation_plot.png
```

Additionally, one can benchark a single configuration with this script using e.g.
```Shell
python latency.py benchmark --device trt --dtype fp16 \
   "(16, 32, 64, 112, 360, 48, 464, 664, 152, 664, 256, 208, 816, 304)"
```

### Network width search
_main file: `train_nas.py`_

_depends on: mxnet, numpy, numba_
* **Train a slimmable network and select a configuration with OWS.** See `-h` for optimization options.
```Shell
python train_nas.py --data /imagenet --latency-target 0.04 \
   --latency-model output/model_trt16_K100.0.jsonl \
   --expname output/ows-trt16-0.04 --resume-last
```
In OWS, the latency target `--latency-target` can be changed during or after training. Using the parameter `--resume-last` allows to resume the last checkpoint without having to retrain, allowing for varying the latency target.

_Implementation detail: for ease of implementation we here use a fixed moving average with a window of `--window=100000` samples for each unary weight, while in the article we used the statistics available over one full last epoch._   

* **Train a slimmable network with AOWS.** See `-h` for optimization options. The outputs, including best configuration for each epochs, are put in the directory corresponding to the parameter `--expname`. 
```Shell
python train_nas.py --data /imagenet --latency-target 0.04 \
   --latency-model output/model_trt16_K100.0.jsonl \
   --AOWS --expname output/aows-trt16-0.04 --resume-last 
```
In AOWS, the latency target `--latency-target` should be set at the beginning of the training, since it impacts the training.


### Training the final model
_main file: `train_final.py`_

_depends on: [mxnet](https://mxnet.apache.org/)_

modified version of gluon-cv's [train_imagenet.py](https://github.com/dmlc/gluon-cv/blob/18f8ab526ffb97660e6e5661f991064c20e2699d/scripts/classification/imagenet/train_imagenet.py) for training mobilenet-v1 with varying channel numbers. Refer to gluon-cv's documentation for detailed usage.

Example command:
```
python train_final.py \
   --rec-train /imagenet/imagenet_train.rec \
   --rec-train-idx /imagenet/imagenet_train.idx \ 
   --rec-val /ramdisk/imagenet_val.rec \
   --rec-val-idx /ramdisk/imagenet_val.idx \
   --use-rec --mode hybrid --lr 0.4 --lr-mode cosine \
   --num-epochs 200 --batch-size 256 -j 32 --num-gpus 4 \
   --dtype float16 --warmup-epochs 5 --no-wd \
   --label-smoothing --mixup \
   --save-dir params_mymobilenet --logging-file mymobilenet.log \
   --configuration "(16, 32, 64, 112, 360, 48, 464, 664, 152, 664, 256, 208, 816, 304)"
```

## Citation
```BibTeX
@InProceedings{Berman2020AOWS,
  author    = {Berman, Maxim and Pishchulin, Leonid and Xu, Ning and Blaschko, Matthew B. and Medioni, Gerard},
  title     = {{AOWS}: adaptive and optimal network width search with latency constraints},
  booktitle = {Proceedings of the {IEEE} Computer Society Conference on Computer Vision and Pattern Recognition},
  month     = jun,
  year      = {2020},
}
```

## Disclaimer
The code was re-implemented and is not fully tested at this point.

