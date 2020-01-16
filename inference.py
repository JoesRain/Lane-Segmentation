from model.deeplabv3plus import DeeplabV3Plus
from config import Config
import torch
import torch.nn as nn
import os
import cv2
from utils.image_process import crop_resize_data
from utils.process_label import decode_color_labels

import numpy as np

device_id = 2

def load_model(model_path):
    """Constructs a atrous deeplabv3plus model."""
    lane_config = Config()
    model = DeeplabV3Plus(lane_config)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda(device=device_id)
        map_location = 'cuda:%d' %device_id
    else:
        map_location = 'cpu'
    model_param = torch.load(model_path,map_location=map_location)['state_dict']
    model_param = {k.replace('module.','') : v for k, v in model_param.items() if (k in model_dict)}
    model.load_state_dict(model_param)
    return model

def img_transform(img):
    img = crop_resize_data(img)
    img = np.transpose(img,(2,0,1))
    img = img[np.newaxis,:].astype(np.float16)
    img = torch.from_numpy(img.copy())
    if torch.cuda.is_available():
        img = img.cuda(device=device_id)
    return img

def get_color_mask(pred):
    pred = torch.softmax(pred,dim=1)
    pred = torch.argmax(pred,dim=1)
    pred = torch.squeeze(pred)
    pred = pred.detach().cpu.numpy()
    pred = decode_color_labels(pred)
    pred = np.transpose(pred,(1,2,0))
    return pred

def inference():
    test_dir = '.'
    lane_config = Config()
    model_path = os.path.join(lane_config.SAVE_PATH,'finalNet.pth.tar')
    model = load_model(model_path)
    image_path = os.path.join(test_dir,'test.jpg')
    img = cv2.imread(image_path)
    img = img_transform(img)
    pred = model(img)
    color_mask = get_color_mask(pred)
    cv2.imwrite(os.path.join(test_dir,'color_mask.jpg'),color_mask)

    return out

