waifu2x iOS Custom Model Specification (v1)
===

Everything described in this document is **case sensitive**.

## File structure
A custom model is a regular zip archive with file extension `.wifm`. It has the following structure:
```
custom_model.wifm
  - manifest.json
  - info.md (optional)
  - main.mlpackage
  - alpha.mlpackage (optional)
```
Note that all files are under root directory inside the `.wifm` archive.

### `manifest.json`
Example:
```json
{
    "version": 1,
    "name": "Custom 2x Model",
    "type": "coreml",
    "subModels": {
        "main": {
            "file": "main.mlmodel",
            "inputName": "input_1",
            "outputName": "Identity"
        },
    },
    "dataFormat": "nchw",
    "inputShape": [1, 3, 128, 128],
    "shrinkSize": 16,
    "scale": 2,
    "alphaMode": "sameAsMain"
}
```

Another example with separate alpha channel model:
```json
{
    "version": 1,
    "name": "Custom 2x Model with Separate Alpha",
    "type": "coreml",
    "subModels": {
        "main": {
            "file": "main.mlpackage",
            "inputName": "input_1",
            "outputName": "Identity"
        },
        "alpha": {
            "file": "alpha.mlpackage",
            "inputName": "input_1",
            "outputName": "Identity"
        }
    },
    "dataFormat": "nchw",
    "inputShape": [1, 3, 128, 128],
    "shrinkSize": 16,
    "scale": 2,
    "alphaMode": "separateModel"
}
```

#### Keys
- `version`: Version of the manifest. Currently the only available version is `1`.
- `name`: Human-readable name of the model. 5-100 characters. Also used to identify the model.
- `type`: Type of the model. Currently the only available type is `coreml`.
- `subModels`: Supported types of submodels are: `main`, `alpha`. At least a `main` model is required.
    - `file`: Path of the submodel file inside the archive with extension. For `coreml` supported file types are: `mlmodel`, `mlpackage`.
    - `inputName`: Name of the input tensor.
    - `outputName`: Name of the output tensor.
- `dataFormat`: Data format of both input and output tensors. Supported values are: `nchw`, `chw`
- `inputShape`: Shape of the input tensor. Width and height must be equal.
- `shrinkSize`: Size to shrink (unstable region) on all 4 sides of input tensor. Applied to output tensor after model inference.
- `scale`: Scale factor of the model. Must be an integer.
- `alphaMode`: Supported values are: `sameAsMain`, `separateModel`. To speed up processing of transparent PNGs you can add a lightweight model as `alpha` submodel in `subModels`, then set this key to `separateModel`.

### Model Files
If your model hasn't been converted to Core ML, refer to [coremltools doc](https://coremltools.readme.io/docs) to convert it.

### `info.md`
An optional Markdown file to include any additional info about the model. [CommonMark](https://commonmark.org/) specification is supported. It will be accessible from the app UI after importing, which is rendered into HTML by [cmark](https://github.com/commonmark/cmark). The best practice is to at least include:
- Use of the model
- Source (author) of the model
- License of the model

The maximum allowed size of `info.md` is 1MB. The encoding must be UTF-8.

## Common Issues

### Tiling
Many models using GAN architecture are prone to tiling artifacts, which are sometimes visible. Unfortunately, since this app mainly targets mobile users with limited RAM, tiling is required. Increasing tile size usually doesn't resolve the issue (sometimes even makes it worse). But you can still try to find the best tile size by tweaking `inputShape`. Note that your model may no longer utilize ANE (Apple Neural Engine) with some unusual tile sizes.

That being said, the built-in [Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) models are implemented using seamless tiling technique and do not have tiling artifacts at all. If there are new models based on this architecture coming out in the future, please send a feedback or create an issue so that we can figure out how to support them in the app as well as this converter.

### Shrink Size
Due to lack of information around edges of tiles, it's important to discard pixels around edges. `shrinkSize` is used to do exactly this. It's applied to input tensors on all 4 edges. Therefore the output square tile size can be calculated as:
```
outputSideLength = (inputSideLength - 2 * shrinkSize) * scale
```
The actual shrinking is performed after model inference. Therefore make sure that your model does not shrink edges by itself. If you have a model that does this, you should append an operation to expand it so that `outputTensorSideLength = inputTensorSideLength * scale`. For example, you can use [pad](https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html) in PyTorch.

### Model Format for Core ML
Both the new `mlpackage` and the old `mlmodel` formats of Core ML models are supported. Generally it's recommended using `mlpackage` format for new models. The app will compile the models and cache them as long as the device storage and OS permits.

### Precision
Only float32 precision is supported on both input and output tensor. Intermediate layers can use anything, which is by default float16 in Core ML models. Please refer to [Typed Execution](https://coremltools.readme.io/docs/typed-execution) to learn how to set Core ML model precision to float32. Generally float16 is good enough with hardly visible degradation in image quality. Using float32 guarantees that your model cannot utilize ANE at all, which is a huge performance loss on ANE-equipped devices.

### Monochrome Models
Currently only models with 3 channels (RGB) are supported. However, it doesn't mean you cannot use 1 channel models. You can convert them into 3 channels. An example of PyTorch implementation used in [converter.py](./converter.py):
```python
import torch
from torch import nn
class MonochromeWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super(MonochromeWrapper, self).__init__()
        self.model = model
    def forward(self, x: torch.Tensor):
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.model(x)
        x = x.repeat([1, 3, 1, 1])
        return x
torch_model = MonochromeWrapper(torch_model)
```
Note that the app can load monochrome images but cannot save them. So output images will always be in RGB(A) channels. It should have almost no impact on model performance as the actual inference is still in monochrome.

### Performance Optimization
Generally ANE (Apple Neural Engine) should be used as much as possible to maximize performance. However, whether a model can utilize ANE varies by a lot of factors, including but not limited to model structure, device, OS version. Fortunately, XCode 14+ added a tool to test if a model is utilizing ANE on a certain device. Please watch [this WWDC 2022 video](https://developer.apple.com/videos/play/wwdc2022/10027/) to learn more.

## Distribution
Make sure you comply with the original model's license/terms if you want to re-distribute the converted custom model. Ask for explicit permission when necessary.
