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

import hashlib
import logging
import os
import subprocess
import sys
import yaml

OUT_DIR = 'out_models'

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

known_models_yaml = None
with open('known_models.yaml', 'r') as f:
    known_models_yaml = yaml.load(f.read(), Loader=yaml.Loader)

num_converted = 0
for model in known_models_yaml['models']:
    file = model['file']
    if not os.path.exists(file):
        continue
    name = model['name']
    out_file = OUT_DIR + '/' + name + '.wifm'
    if os.path.exists(out_file):
        logger.warning('Skipping already converted model %s', name)
        continue
    logger.info('Converting model: %s', name)
    with open(file, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
        if digest != model['sha256']:
            logger.error('Unexpected hash for file %s, skipping', file)
            continue
    command = [
        'python', 'converter.py',
        '--type', model['type'],
        '--name', name,
        '--scale', str(model['scale']),
        '--out-dir', OUT_DIR,
        '--description', model['description'],
        '--source', model['source'],
        '--author', model['author'],
        '--license', model['license']
    ]
    if 'cuda' in model and model['cuda']:
        command += ['--has-cuda']
    if 'monochrome' in model and model['monochrome']:
        command += ['--monochrome']
    if 'features' in model:
        command += ['--num-features', str(model['features'])]
    if 'blocks' in model:
        command += ['--num-blocks', str(model['blocks'])]
    if 'convs' in model:
        command += ['--num-convs', str(model['convs'])]
    if 'shuffle-factor' in model:
        command += ['--shuffle-factor', str(model['shuffle-factor'])]
    command += [file]
    logger.debug('Command: %s', command)
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    for c in iter(lambda: process.stdout.read(1), b''):
        sys.stdout.buffer.write(c)
        sys.stdout.flush()
    process.communicate()
    if process.returncode != 0:
        break
    num_converted += 1

if num_converted > 0:
    logger.info('Converted %d out of %d supported models', num_converted, len(known_models_yaml['models']))
    logger.info('Output models are saved to %s directory', OUT_DIR)
