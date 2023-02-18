waifu2x iOS Model Converter
===

## Introduction
This is a ML model converter for [waifu2x iOS](https://apps.apple.com/app/apple-store/id1286485858) custom model feature.

## Pre-converted models
A few models from [upscale.wiki](https://upscale.wiki/wiki/Model_Database) with permissive licenses and are available for download:
- [Google Drive](https://drive.google.com/drive/folders/1btfOExWcbO3qTN2uad61k2T0hBjCM0tv?usp=share_link)
- [百度网盘](https://pan.baidu.com/s/1KFuncLytPdSMC_xMJ3l9dw?pwd=8aoz) (For users based in China)

If another model you want has a permissive license for commercial use (CC0, MIT, Unlicense, etc.) and a supported architecture (see below), you can request a conversion by creating an issue. Otherwise please convert on your own. Note that models without a license are not permissive. **Also note that this will be handled in best-effort basis without any promise and SLO.**

## Custom Model Specification
There are features not supported by this converter but are supported by the app. See [SPECS.md](./SPECS.md) if you want to create a custom model package from scratch and make use of advanced features.

## Supported Architectures
- [ESRGAN ("old arch")](https://github.com/xinntao/ESRGAN/tree/old-arch), including lite version
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), including compact version (SRVGGNetCompact)

## System Requirements
- macOS or x86_64 Linux
- GPU supporting [MPS](https://pytorch.org/docs/stable/notes/mps.html) or CUDA (not required but speeds up conversion)
- Python 3

Tested environments: macOS 13.1 / Debian 11, Python 3.10.9 (conda 23.1.0)

## Installation
It's recommended that you create a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) or [virtualenv](https://virtualenv.pypa.io/en/latest/) so that dependencies installed later will be the correct (tested) version.

```bash
git clone --recursive https://github.com/imxieyi/waifu2x-ios-model-converter.git
cd waifu2x-ios-model-converter
pip install -r requirements.txt
```

## Usages

### Single Model Conversion
To convert a single model, use [converter.py](./converter.py). Please run the following command for help on how to use:
```bash
python converter.py --help
```

### Batch Conversion for Tested Models
There are some known working models (from [upscale.wiki](https://upscale.wiki/wiki/Model_Database)) defined in [known_models.yaml](./known_models.yaml). To convert a subset or all of them, place downloaded `.pth` files under [torch_models](./torch_models). Then you can run `python batch_convert.py` to convert them. Resulting models will be placed under [out_models](./out_models)

If there is a model you have problem converting, please create an issue so that it can be investigated. **There is no guarantee that your problem will be solved due to the complexity of model conversion.** Note that models with a strict license for commercial use (GNU GPLv3, CC BY-NC 4.0, etc.) must be converted by yourself.
