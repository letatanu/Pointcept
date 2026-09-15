"""OPTIONAL trainer adapter. The standalone network does not import this file.

Copy the complete aerial_wavelet_standalone directory into pointcept/models/.
Import this adapter from pointcept/models/__init__.py to register the segmentor.
Only the training registry/loss builder is reused; no Pointcept backbone code.
"""
from torch import nn
from pointcept.models.builder import MODELS
from pointcept.models.losses import build_criteria
from pointcept.models.ptno import AerialWaveletNet


@MODELS.register_module('AerialWaveletSegmentor')
class AerialWaveletSegmentor(nn.Module):
    def __init__(self, network, criteria):
        super().__init__()
        self.network=AerialWaveletNet(**network)
        self.criteria=build_criteria(criteria)

    def forward(self, input_dict):
        logits=self.network(input_dict)
        if self.training:
            return dict(loss=self.criteria(logits,input_dict['segment']))
        result=dict(seg_logits=logits)
        if 'segment' in input_dict:
            result['loss']=self.criteria(logits,input_dict['segment'])
        return result
