#!/usr/bin/env python3

# Copyright 2023 Yi Xie
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import platform
import os
import shutil
import sys
import zipfile

parser = argparse.ArgumentParser(
    prog=os.path.basename(__file__),
    description='Convert a ML model to waifu2x app custom model',
)
parser.add_argument('filename')
required_args = parser.add_argument_group('required')
required_args.add_argument('--type', choices=['esrgan_old', 'esrgan_old_lite', 'real_esrgan', 'real_esrgan_compact'], required=True, help='Type of the model')
required_args.add_argument('--name', type=str, required=True, help='Name of the model')
required_args.add_argument('--scale', type=int, required=True, help='Scale factor of the model')
required_args.add_argument('--out-dir', type=str, required=True, help='Output directory')
optional_args = parser.add_argument_group('optional')
optional_args.add_argument('--monochrome', action='store_true', help='Input model is monochrome (single channel)')
optional_args.add_argument('--has-cuda', action='store_true', help='Input model contains CUDA object')
optional_args.add_argument('--input-size', type=int, default=256, help='Input size (both width and height), default to 256')
optional_args.add_argument('--shrink-size', type=int, default=20, help='Shrink size (applied to all 4 sides on input), default to 20')
optional_args.add_argument('--description', type=str, required=False, help='Description of the model, supports Markdown')
optional_args.add_argument('--source', type=str, required=False, help='Source of the model, supports Markdown')
optional_args.add_argument('--author', type=str, required=False, help='Author of the model, supports Markdown')
optional_args.add_argument('--license', type=str, required=False, help='License of the model, supports Markdown')
optional_args.add_argument('--info-md', type=str, required=False, help='Use custom info.md instead of individual flags')
optional_args.add_argument('--no-delete-mlmodel', action='store_true', help='Don\'t delete the intermediate Core ML model file')
args = parser.parse_args()

logger = logging.getLogger('converter')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

if args.input_size % 4 != 0:
    logger.fatal('Input size must be multiple of 4')
    sys.exit(-1)

if args.shrink_size < 0:
    logger.fatal('Shrink size must not be < 0')
    sys.exit(-1)

if args.input_size - 2 * args.shrink_size < 4:
    logger.fatal('Input size after shrinking is too small')
    sys.exit(-1)

os.makedirs(args.out_dir, exist_ok=True)

import coremltools as ct
import psutil
import torch

torch_model = None
input_tensor = None
output_tensor = None

device = torch.device('cpu')
if platform.system() == 'Darwin' and torch.backends.mps.is_available():
    device = torch.device('mps')
    logger.info('Using torch device mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
    logger.info('Using torch device cuda')
else:
    logger.info('Using torch device cpu, please be patient')

logger.info('Creating model architecture')
channels = 3
if args.monochrome:
    channels = 1
if args.type == 'esrgan_old':
    from esrgan_old import architecture
    torch_model = architecture.RRDB_Net(
        channels, channels, 64, 23, gc=32, upscale=args.scale, norm_type=None,
        act_type='leakyrelu', mode='CNA', res_scale=1, upsample_mode='upconv')
elif args.type == 'esrgan_old_lite':
    from esrgan_old import architecture
    torch_model = architecture.RRDB_Net(
        channels, channels, 32, 12, gc=32, upscale=args.scale, norm_type=None,
        act_type='leakyrelu', mode='CNA', res_scale=1, upsample_mode='upconv')
elif args.type == 'real_esrgan':
    from basicsr.archs.rrdbnet_arch import RRDBNet
    torch_model = RRDBNet(num_in_ch=channels, num_out_ch=channels, num_feat=64, num_block=23, num_grow_ch=32, scale=args.scale)
elif args.type == 'real_esrgan_compact':
    from basicsr.archs.srvgg_arch import SRVGGNetCompact
    torch_model = SRVGGNetCompact(num_in_ch=channels, num_out_ch=channels, num_feat=64, num_conv=16, upscale=args.scale, act_type='prelu')
else:
    logger.fatal('Unknown model type: %s', args.type)
    sys.exit(-1)

logger.info('Loading weights')
loadnet = None
if args.has_cuda:
    loadnet = torch.load(args.filename, map_location=device)
else:
    loadnet = torch.load(args.filename)

if 'params_ema' in loadnet:
    loadnet = loadnet['params_ema']
elif 'params' in loadnet:
    loadnet = loadnet['params']
torch_model.load_state_dict(loadnet, strict=True)

if args.monochrome:
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

logger.info('Tracing model, will take a long time and a lot of RAM')
torch_model.eval()
torch_model = torch_model.to(device)
example_input = torch.zeros(1, 3, 64, 64)
example_input = example_input.to(device)
traced_model = torch.jit.trace(torch_model, example_input)
out = traced_model(example_input)
logger.info('Successfully traced model')

input_size = example_input.shape[-1]
output_size = out.shape[-1]
if args.scale != output_size / input_size:
    logger.fatal('Expected output size to be %d, but is actually %.2f', args.scale, output_size / input_size)
    sys.exit(-1)

logger.info('Converting to Core ML')
input_shape = [1, 3, args.input_size, args.input_size]
output_size = args.input_size * args.scale
output_shape = [1, 3, output_size, output_size]
model = ct.convert(
    traced_model,
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=input_shape)]
)
model_name = args.filename.split('/')[-1].split('.')[0]
mlmodel_file = args.out_dir + '/' + model_name + '.mlpackage'
model.save(mlmodel_file)

logger.info('Packaging model')
spec = model.get_spec()
input_name = spec.description.input[0].name
output_name = spec.description.output[0].name
logger.debug('Model input name: %s, size: %s', input_name, args.input_size)
output_size_shrinked = (args.input_size - 2 * args.shrink_size) * args.scale
logger.debug('Model output name: %s, size: %s, after shrinking: %s', output_name, output_size, output_size_shrinked)

manifest = {
    "version": 1,
    "name": args.name,
    "type": "coreml",
    "subModels": {
        "main": {
            "file": mlmodel_file,
            "inputName": input_name,
            "outputName": output_name
        }
    },
    "dataFormat": "nchw",
    "inputShape": input_shape,
    "outputShape": output_shape,
    "shrinkSize": args.shrink_size,
    "scale": args.scale,
    "alphaMode": "sameAsMain"
}

info_md = '''
{}
===
Converted by [waifu2x-ios-model-converter](https://github.com/imxieyi/waifu2x-ios-model-converter).

'''.format(args.name)

if args.description is not None:
    info_md += '''
## Description
{}

'''.format(args.description)

if args.author is not None:
    info_md += '''
## Author
{}

'''.format(args.author)

if args.source is not None:
    info_md += '''
## Source
{}

'''.format(args.source)

if args.license is not None:
    info_md += '''
## License
{}

'''.format(args.license)

if len(info_md) > 1024 * 1024:
    logger.fatal('Model info.md too large. Try to reduce license file size, etc.')
    sys.exit(-1)

def add_folder_to_zip(folder, zipfile):
    for folderName, subfolders, filenames in os.walk(folder):
        for filename in filenames:
            filePath = os.path.join(folderName, filename)
            zipfile.write(filePath, filePath)

zip_file = args.out_dir + '/' + args.name + '.wifm'
with zipfile.ZipFile(zip_file, 'w', compression=zipfile.ZIP_DEFLATED) as modelzip:
    modelzip.writestr('manifest.json', json.dumps(manifest))
    modelzip.writestr('info.md', info_md)
    if os.path.isfile(mlmodel_file):
        modelzip.write(mlmodel_file)
    else:
        add_folder_to_zip(mlmodel_file, modelzip)

if not args.no_delete_mlmodel:
    if os.path.isfile(mlmodel_file):
        os.remove(mlmodel_file)
    else:
        shutil.rmtree(mlmodel_file)

logger.info('Successfully converted model: %s', zip_file)
