"""
Model defining the search space.
Implemented: Mobilenet-v1 family.
"""

import torch
from torch import nn
import numpy as np
from collections import namedtuple
import logging


logger = logging.getLogger(__name__)
LayerType = namedtuple('LayerType', ['in_size', 'kernel_size', 'stride', 'dw', 'bias'])


class Moduler(nn.Module):
    """
    Dynamically trims input channels (randomly or based on argument)
    """

    def __init__(self, configurations):
        super().__init__()
        self.configurations = configurations
        self.base_channels = 8
        self.probability = None  # for biased sampling

    def forward(self, data, channels=None, record=True):
        if not isinstance(data, dict):
            data = dict(x=data)
        x = data['x']

        if channels is None:
            idx = np.random.choice(np.arange(len(self.configurations)),
                                   size=x.size(0),
                                   p=self.probability)
            confs = self.configurations[idx]
        else:
            confs = channels * np.ones((x.size(0),), int)

        mask = x.new_zeros((x.size(0), x.size(1) + 1))
        mask[np.arange(len(confs)), confs] = 1.0
        mask = 1 - mask[:, :x.size(1)].cumsum(1)
        x = x * mask.unsqueeze(2).unsqueeze(3)

        data['x'] = x
        if record:  # record chosen channels
            if 'decision' not in data: data['decision'] = []
            data['decision'].append(confs)
        return data

    def __repr__(self):
        return "Moduler({})".format(self.configurations)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class SlimMobilenet(nn.Module):

    in_channels = 3
    out_channels = 1000
    
    @staticmethod
    def gen_conv(inp, oup, stride, dw=False, bn=True):
        mod = []
        if dw:
            mod = [
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            ]
        else:
            mod = [
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            ]
        if not bn:
            mod = [m for m in mod if not isinstance(m, nn.BatchNorm2d)]
        return nn.Sequential(*mod)
        
    @staticmethod
    def strides_channels():
        """
        follows Mobilenet-v1 definition
        """
        blocks = [[32, 64],  # special first stem block
                  [128, 128],
                  [256, 256],
                  [512, 512, 512, 512, 512, 512],
                  [1024, 1024],
                  ]
        strides = []
        for block in blocks:
            strides.extend([2] + [1] * (len(block) - 1))
        base_channels = np.array([c for block in blocks for c in block])
        
        return strides, base_channels
    
    def __init__(self, min_width=0.2, max_width=1.5, levels=14, fc_dropout=0.0, in_size=(224, 224)):
        super().__init__()

        def divise8(i):
            return (np.maximum(np.round(i / 8), 1) * 8).astype(int)

        strides, base_channels = self.strides_channels()
        depthwise = [0] + [1] * (len(base_channels) - 1)

        self.configurations = divise8(base_channels.reshape(-1, 1) * np.linspace(min_width, max_width, levels).reshape(1, -1))
        self.configurations = [np.unique(c) for c in self.configurations]

        self.components = []

        channels = [self.in_channels] + [int(c[-1]) for c in self.configurations]
        inp = iter(channels)
        oup = iter(channels[1:])

        self.model = nn.ModuleList()
        for dw, strid in zip(depthwise, strides):
            I = next(inp)
            O = next(oup)
            mod = self.gen_conv(I, O, strid, dw)
            component = LayerType(in_size=in_size, kernel_size=3, stride=strid, dw=bool(dw), bias=False)
            in_size = (in_size[0] // strid, in_size[1] // strid)
            self.model.append(mod)
            self.components.append(component)

        self.filters = nn.ModuleList()
        for conf, base_chan in zip(self.configurations, base_channels):
            F = Moduler(conf)
            F.base_channels = base_chan
            self.filters.append(F)

        self.pool = nn.AvgPool2d(7)
        self.fc_dropout = None if not fc_dropout else nn.Dropout(fc_dropout)
        in_size = (in_size[0] // 7, in_size[1] // 7)
        
        I = next(inp)
        self.fc = nn.Linear(I, self.out_channels)
        self.components.append(LayerType(in_size=in_size, kernel_size=1, stride=1, dw=False, bias=True))

    def forward(self, data, configuration=None):
        if not isinstance(data, dict):
            data = dict(x=data)
        for i, (conv, filter) in enumerate(zip(self.model, self.filters)):
            data['x'] = conv(data['x'])
            data = filter(data,
                          channels=(configuration[i] if configuration is not None else None))
        data['x'] = self.pool(data['x'])
        data['x'] = data['x'].view(data['x'].size(0), -1)
        if self.fc_dropout is not None:
            data['x'] = self.fc_dropout(data['x'])
        data['x'] = self.fc(data['x'])
        data['decision'] = torch.tensor(np.array(data['decision']).T, device=data['x'].device)
        return data
    
    @classmethod
    def reduce(cls, C=(32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024),
               bn=True):
        """
        Remove all modulers and reduce according to a single channel configuration
        """
        modules = []
        I = cls.in_channels
        depthwise = [False] + [True] * (len(C) - 1)
        strides, base_channels = cls.strides_channels()
        assert len(strides) == len(C)
        
        for O, stride, dw in zip(C, strides, depthwise):
            modules.append(cls.gen_conv(I, O, stride, dw, bn=bn))
            I = O
        modules += [nn.AvgPool2d(7), Flatten(), nn.Linear(I, cls.out_channels)]
        reduced = nn.Sequential(*modules)
        return reduced
    

