"""Load released JaNai checkpoints using Spandrel's architecture detection."""

from torch import nn
from spandrel import ImageModelDescriptor, ModelLoader


class RGBModel(nn.Module):
    """Adapt monochrome models to the app's RGB, [0, 1] tensor contract."""

    def __init__(self, descriptor):
        super().__init__()
        self.model = descriptor.model
        self.monochrome = descriptor.input_channels == 1

    def forward(self, x):
        if self.monochrome:
            x = x.mean(dim=1, keepdim=True)
        x = self.model(x)
        if self.monochrome:
            x = x.repeat(1, 3, 1, 1)
        return x.clamp(0, 1)


def load_spandrel_model(filename):
    descriptor = ModelLoader(device='cpu').load_from_file(filename)
    if not isinstance(descriptor, ImageModelDescriptor):
        raise ValueError('Only image-to-image models are supported')
    if descriptor.architecture.id not in {'ESRGAN', 'DAT', 'FDAT', 'HAT', 'SPAN'}:
        raise ValueError(f'Untested architecture: {descriptor.architecture.id}')
    if descriptor.input_channels != descriptor.output_channels or descriptor.input_channels not in (1, 3):
        raise ValueError('The app requires RGB or single-channel image models')
    # Release archives contain fp16/bf16 weights. Promote them before tracing;
    # Core ML chooses intermediate precision separately from tensor I/O types.
    descriptor.model.float().eval()
    return RGBModel(descriptor).eval(), descriptor
