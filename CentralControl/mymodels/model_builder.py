# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:06:58 2024

@author: ab19109
CNNを呼び出すプログラム(Torchreidのbuild_model参考)
"""
from __future__ import division, print_function, absolute_import
import pickle
import warnings
from functools import partial
import os.path as osp
import torch
from collections import OrderedDict

from .myosnet_highres1 import osnet_x1_0 as osnet_highres1
from .myosnet_highres2 import osnet_x1_0 as osnet_highres2
from .osnet_base import osnet_x1_0 as osnet
from .osnet_part_addblock import osnet_x1_0 as osnet_part_addblock
from .osnet_part_addblock_dellarge import osnet_x1_0 as osnet_part_addblock_dellarge
from .osnet_part_delsmall import osnet_x1_0 as osnet_part_delsmall




model_container = {
    'osnet': osnet,
    'osnet_highres1': osnet_highres1,
    'osnet_highres2': osnet_highres2,
    'osnet_part_addblock': osnet_part_addblock,
    'osnet_part_addblock_dellarge': osnet_part_addblock_dellarge,
    'osnet_part_delsmall': osnet_part_delsmall
    }


'''
CNNのクラスのオブジェクトを生成する関数(Torchreid参考)
'''
def build_model(name, num_classes=1, loss='softmax', pretrained=True, use_gpu=True):
    '''
    Parameters
    ----------
    name : str
        CNNの名前
    num_classes : int, optional
        分類クラス数．The default is 1
    loss : str, optional
        損失関数． The default is 'softmax'.
    pretrained : bool, optional
        ImageNetによる事前学習済モデルを用いるか． The default is True.
    use_gpu : bool, optional
        GPUを使うか． The default is True.

    Raises
    ------
    KeyError
        指定されたCNNの名前がmodel_containerに含まれていない場合のメッセージ

    Returns
    -------
    model : nn.Module
        CNN

    '''
    
    
    model_set = list(model_container.keys())
    if name not in model_set:
        raise KeyError(
            'Unknown model: {}. Model must be one of {}'.format(name, model_set)
            )
    
    model = model_container[name](
        num_classes = num_classes,
        loss = loss,
        pretained = pretrained,
        use_gou = use_gpu
        )
    
    return model



'''
学習済みモデルを読み込む関数
'''
def load_model(model, weight_path):
    '''
    Parameters
    ----------
    model : nn.Module
        CNN
    weight_path : str
        パラメータファイルのパス

    Returns
    -------
    None.

    '''
    
    #Load checkpoint
    if weight_path is None:
        raise ValueError("File path is None")
    
    weight_path = osp.abspath(osp.expanduser(weight_path))
    if not osp.exists(weight_path):
        raise FileNotFoundError("File is not found at <{}>".format(weight_path))
        
    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(weight_path, map_location=map_location)
        
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding='latin1')
        pickle.Unpickler = partial(pickle.Unpickler, encoding='latin1')
        checkpoint = torch.load(weight_path, pickle_module=pickle, map_location=map_location)
    
    except Exception:
        print("Unable to load checkpoint from <{}>".format(weight_path))
        raise
        
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        
    else:
        state_dict = checkpoint
    
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers = []
    discarded_layers = []
    
    for k, v in state_dict.items():
        if k.startswith("module."):
            #discard modules.
            k = k[7:]
            
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
            
        else:
            discarded_layers.append(k)
    
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            "The pretrained weights <{}> cannnot be loaded, "
            "please check the key named manually"
            "(** ignored and continue **".format(weight_path)
        )
    
    else:
        print("Successfully loaded pretrained weights from <{}>".format(weight_path))
        if len(discarded_layers) > 0:
            pass
# =============================================================================
#             print("** The following layers are discrded "
#                   "due to unmatched keys or layer size: {}".format(discarded_layers)
#               )
# 
# =============================================================================



