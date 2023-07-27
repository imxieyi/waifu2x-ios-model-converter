waifu2x-ios Model Converter
===

## Introduction
This is a Core ML model converter for [waifu2x-ios (App Store version)](https://apps.apple.com/app/apple-store/id1286485858) custom models feature. Both iOS/iPadOS and macOS versions of the app are supported.

## Pre-converted Models
A few models from [upscale.wiki](https://upscale.wiki/wiki/Model_Database) with permissive licenses are pre-converted and available to download:
- [Google Drive](https://drive.google.com/drive/folders/1btfOExWcbO3qTN2uad61k2T0hBjCM0tv?usp=share_link)
- [百度网盘](https://pan.baidu.com/s/1KFuncLytPdSMC_xMJ3l9dw?pwd=8aoz) (For users based in China)

If another model you want has a permissive license for commercial use (CC0, MIT, Unlicense, etc.) and a supported architecture (see below), you can request a conversion by creating an issue. Note that models without any license are not permissive. **Also note that this is not a service, so don't set any expectations here.**

## Web Converter
- https://huggingface.co/spaces/imxieyi/waifu2x-ios-model-converter

The web converter is hosted on [Hugging Face Spaces](https://huggingface.co/spaces) to convert tested models defined in [known_models.yaml](./known_models.yaml).

## Colab Converter
[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imxieyi/waifu2x-ios-model-converter/blob/master/esrgan_to_waifu2x_ios.ipynb)

In case the web converter doesn't work, please use the Colab converter instead.

## Custom Model Specification
You don't have to use this tool to create custom models. See [SPECS.md](./SPECS.md) if you want to create one from scratch.

## Supported Architectures
- [ESRGAN ("old arch")](https://github.com/xinntao/ESRGAN/tree/old-arch), including lite version
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), including compact version (SRVGGNetCompact)
- [ESRGAN+ (nESRGAN+)](https://github.com/ncarraz/ESRGANplus)

## System Requirements
- macOS or x86_64 Linux
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

### Batch Conversion of Tested Models
There are some known working models (from [upscale.wiki](https://upscale.wiki/wiki/Model_Database)) defined in [known_models.yaml](./known_models.yaml). To convert a subset or all of them, place downloaded `.pth` files under [torch_models](./torch_models). Then you can run `python batch_convert.py` to convert them. Resulting models will be placed under [out_models](./out_models)

If there is a model you have trouble converting, please create an issue so that it can be investigated. **There is no guarantee that your problem will be solved due to the complex nature of model conversion.** Note that models with a strict license for commercial use (GNU GPLv3, CC BY-NC 4.0, etc.) must be converted by yourself eventually.

## Troubleshoot
### `BlobWriter not loaded` Error
This happens because you are running the script on an unsupported platform. Unfortunately [coremltools pip package](https://pypi.org/project/coremltools/#files) only supports macOS and x86_64 Linux. If you are using Windows please try [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/).

## Issue Reporting
Privacy is one of the highest priorities of the app. So all imported models will only be kept locally on your device (except iCloud device backup, of course). We will have no enough information for debugging if the app throws a random error or crashes when you try to import a shiny new custom model you just created. Therefore please provide a sample model (and the architecture definition code) when you report issues.
