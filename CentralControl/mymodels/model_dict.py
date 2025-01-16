# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:19:45 2024

@author: ab19109
CNNをまとめた辞書を管理する辞書を作る
"""
from .model_builder import build_model, load_model



class ModelDict:
    def __init__(self):

        self.model_dict = {
            'wholebody': {
                'path': r'E:\md23036\Model\best_models\wholebody\model.pth.tar-22',
                'model_name': 'osnet_highres1',
                'size': (512, 256)
                },
            'face': {
                'weight': 2.0,
                'path': r'E:\md23036\Model\best_models\part\model_face_3.pth.tar-4',
                'model_name': 'osnet',
                'size': (256, 128)
                },
            'back_head': {
                'weight': 0.25,
                'path': r'E:\md23036\Model\best_models\part\model_backhead_1.pth.tar-8',
                'model_name': 'osnet',
                'size': (256, 128)
                },
            'chest': {
                'weight': 0.25,
                'path': r'E:\md23036\Model\best_models\part\model_chest_addblock_dellarge_2.pth.tar-24',
                'model_name': 'osnet_part_addblock_dellarge',
                'size': (256, 128)
                },
            'back': {
                'weight': 0.25,
                'path': r'E:\md23036\Model\best_models\part\model_back_5.pth.tar-22',
                'model_name': 'osnet',
                'size': (256, 128)
                },
            'right_arm': {
                'weight': 1.0,
                'path': r'E:\md23036\Model\best_models\part\model_right_arm_2.pth.tar-2',
                'model_name': 'osnet',
                'size': (256, 128)
                },
            'right_wrist': {
                'weight': 1.5,
                'path': r'E:\md23036\Model\best_models\part\model_right_wrist_delsmall_5.pth.tar-24',
                'model_name': 'osnet_part_delsmall',
                'size': (256, 128)
                },
            'left_arm': {
                'weight': 1.0,
                'path': r'E:\md23036\Model\best_models\part\model_left_arm_4.pth.tar-25',
                'model_name': 'osnet',
                'size': (256, 128)
                },
            'left_wrist': {
                'weight': 1.5,
                'path': r'E:\md23036\Model\best_models\part\model_left_wrist_3.pth.tar-5',
                'model_name': 'osnet',
                'size': (256, 128)
                },
            'leg': {
                'weight': 0.1,
                'path': r'E:\md23036\Model\best_models\part\model_leg_4.pth.tar-6',
                'model_name': 'osnet',
                'size': (256, 128)
                },
            'right_foot': {
                'weight': 2.0,
                'path': r'E:\md23036\Model\best_models\part\model_right_foot_resize_5.pth.tar-19',
                'model_name': 'osnet',
                'size': (64, 128)
                },
            'left_foot': {
                'weight': 2.0,
                'path': r'E:\md23036\Model\best_models\part\model_left_foot_resize_2.pth.tar-23',
                'model_name': 'osnet',
                'size': (64, 128)
                }
            }


    def get_item(self, separate=False):
        for part in self.model_dict.keys():            
            model = build_model(
                name = self.model_dict[part]['model_name'] 
            )
            model = model.cuda()
            model.eval()
            
            load_model(model, self.model_dict[part]['path'])
            
            self.model_dict[part]['model'] = model
        
        if separate:
            dict_wholebody = self.model_dict['wholebody']
            del self.model_dict['wholebody']
        
            return dict_wholebody, self.model_dict
    
        else:
            
            return self.model_dict
        
        
        
        
'''
CNN改良前のもの
self.model_dict = {
    'wholebody': {
        'path': r'E:\md23036\Model\best_models\wholebody\model.pth.tar-22',
        'model_name': 'osnet_highres1',
        'size': (512, 256)
        },
    'face': {
        'weight': 2.0,
        'path': r'E:\md23036\Model\best_models\part\model_face_3.pth.tar-4',
        'model_name': 'osnet',
        'size': (256, 128)
        },
    'back_head': {
        'weight': 0.25,
        'path': r'E:\md23036\Model\best_models\part\model_backhead_1.pth.tar-8',
        'model_name': 'osnet',
        'size': (256, 128)
        },
    'chest': {
        'weight': 0.25,
        'path': r'E:\md23036\Model\remodeling\part2\chest\mydata21_osnet_chest_3\model\model.pth.tar-13',
        'model_name': 'osnet',
        'size': (256, 128)
        },
    'back': {
        'weight': 0.25,
        'path': r'E:\md23036\Model\best_models\part\model_back_5.pth.tar-22',
        'model_name': 'osnet',
        'size': (256, 128)
        },
    'right_arm': {
        'weight': 1.0,
        'path': r'E:\md23036\Model\best_models\part\model_right_arm_2.pth.tar-2',
        'model_name': 'osnet',
        'size': (256, 128)
        },
    'right_wrist': {
        'weight': 1.5,
        'path': r'E:\md23036\Model\remodeling\part2\right_wrist\mydata21_osnet_right_wrist_5\model\model.pth.tar-15',
        'model_name': 'osnet',
        'size': (256, 128)
        },
    'left_arm': {
        'weight': 1.0,
        'path': r'E:\md23036\Model\best_models\part\model_left_arm_4.pth.tar-25',
        'model_name': 'osnet',
        'size': (256, 128)
        },
    'left_wrist': {
        'weight': 1.5,
        'path': r'E:\md23036\Model\best_models\part\model_left_wrist_3.pth.tar-5',
        'model_name': 'osnet',
        'size': (256, 128)
        },
    'leg': {
        'weight': 0.1,
        'path': r'E:\md23036\Model\best_models\part\model_leg_4.pth.tar-6',
        'model_name': 'osnet',
        'size': (256, 128)
        },
    'right_foot': {
        'weight': 2.0,
        'path': r'E:\md23036\Model\best_models\part\model_right_foot_resize_5.pth.tar-19',
        'model_name': 'osnet',
        'size': (256, 128)
        },
    'left_foot': {
        'weight': 2.0,
        'path': r'E:\md23036\Model\best_models\part\model_left_foot_resize_2.pth.tar-23',
        'model_name': 'osnet',
        'size': (256, 128)
        }
    }
'''
'''
改良後
self.model_dict = {
    'wholebody': {
        'path': r'E:\md23036\Model\best_models\wholebody\model.pth.tar-22',
        'model_name': 'osnet_highres1',
        'size': (512, 256)
        },
    'face': {
        'weight': 2.0,
        'path': r'E:\md23036\Model\best_models\part\model_face_3.pth.tar-4',
        'model_name': 'osnet',
        'size': (256, 128)
        },
    'back_head': {
        'weight': 0.25,
        'path': r'E:\md23036\Model\best_models\part\model_backhead_1.pth.tar-8',
        'model_name': 'osnet',
        'size': (256, 128)
        },
    'chest': {
        'weight': 0.25,
        'path': r'E:\md23036\Model\best_models\part\model_chest_addblock_dellarge_2.pth.tar-24',
        'model_name': 'osnet_part_addblock_dellarge',
        'size': (256, 128)
        },
    'back': {
        'weight': 0.25,
        'path': r'E:\md23036\Model\best_models\part\model_back_5.pth.tar-22',
        'model_name': 'osnet',
        'size': (256, 128)
        },
    'right_arm': {
        'weight': 1.0,
        'path': r'E:\md23036\Model\best_models\part\model_right_arm_2.pth.tar-2',
        'model_name': 'osnet',
        'size': (256, 128)
        },
    'right_wrist': {
        'weight': 1.5,
        'path': r'E:\md23036\Model\best_models\part\model_right_wrist_delsmall_5.pth.tar-24',
        'model_name': 'osnet_part_delsmall',
        'size': (256, 128)
        },
    'left_arm': {
        'weight': 1.0,
        'path': r'E:\md23036\Model\best_models\part\model_left_arm_4.pth.tar-25',
        'model_name': 'osnet',
        'size': (256, 128)
        },
    'left_wrist': {
        'weight': 1.5,
        'path': r'E:\md23036\Model\best_models\part\model_left_wrist_3.pth.tar-5',
        'model_name': 'osnet',
        'size': (256, 128)
        },
    'leg': {
        'weight': 0.1,
        'path': r'E:\md23036\Model\best_models\part\model_leg_4.pth.tar-6',
        'model_name': 'osnet',
        'size': (256, 128)
        },
    'right_foot': {
        'weight': 2.0,
        'path': r'E:\md23036\Model\best_models\part\model_right_foot_resize_5.pth.tar-19',
        'model_name': 'osnet',
        'size': (64, 128)
        },
    'left_foot': {
        'weight': 2.0,
        'path': r'E:\md23036\Model\best_models\part\model_left_foot_resize_2.pth.tar-23',
        'model_name': 'osnet',
        'size': (64, 128)
        }
    }
'''
