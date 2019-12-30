"""
Author: Jingxiao Gu
Baidu Account: Seigato
Description: Encode & Decode Labels for Lane Segmentation Competition
NOTE: For 8 classes
"""

import numpy as np

def encode_labels(color_mask):
    encode_mask = np.zeros((color_mask.shape[0], color_mask.shape[1]))
    encode_dict = {0:[0,249,255,213,206,207,211,208,216,215,218,219,232,202,231,230,228,229,233,212,223,249,255],1:[200,204,209],2:[201,203],
                   3:[217],4:[210],5:[214],6:[220,221,222,224,225,226],
                   7:[205,227,250]}
    for key in encode_dict.keys():
        for color in encode_dict[key]:
            encode_mask[color_mask == color] = key
    return encode_mask


def decode_labels(labels):
    deocde_mask = np.zeros((labels.shape[0], labels.shape[1]), dtype='uint8')
    encode_dict = {0: 0, 1: 200, 2: 201,3: 217,
                   4: 210, 5: 214,6: 220,7: 205}
    for key in encode_dict.keys():
        deocde_mask[labels==key] = encode_dict[key]
    return deocde_mask

def decode_color_labels(labels):
    decode_mask = np.zeros((3, labels.shape[0], labels.shape[1]), dtype='uint8')
    encode_dict = {1: [0, 0, 0], 1: [220, 20, 60], 2: [119, 11, 32],
                   3: [220, 220, 0], 4: [128,64, 128],
                   5: [190,153,153],6: [128, 128, 0],7: [178, 132, 190]}
    for key in encode_dict.keys():
        for i in range(3):
            decode_mask[i][labels==key] = encode_dict[key][i]
    return decode_mask

def verify_labels(labels):
    pixels = [0]
    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            pixel = labels[x, y]
            if pixel not in pixels:
                pixels.append(pixel)
    print('The Labels Has Value:', pixels)
