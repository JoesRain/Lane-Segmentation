from model.deeplabv3plus import DeeplabV3Plus
from config import Config
import torch
import pandas as pd
import os
import cv2
from utils.image_process import crop_resize_data
from utils.process_labels import decode_color_labels

import numpy as np

device_id = 0


def load_model(model_path):
    """Constructs a atrous deeplabv3plus model."""
    lane_config = Config()
    model = DeeplabV3Plus(lane_config)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda(device=device_id)
    checkpoint = torch.load(model_path)
    model_param = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(model_param)
    return model


def img_transform(img):
    img = crop_resize_data(img)
    img = np.transpose(img, (2, 0, 1))
    img = img[np.newaxis, :].astype(np.float32)
    img = torch.from_numpy(img.copy())
    if torch.cuda.is_available():
        img = img.cuda(device=device_id)
    return img


def get_color_mask(pred):
    pred = torch.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)
    pred = torch.squeeze(pred)
    print(pred)
    pred = pred.detach().data.cpu().numpy()
    pred = decode_color_labels(pred)
    pred = np.transpose(pred, (1, 2, 0))
    return pred


def inference():
    test_dir = '.'
    lane_config = Config()
    model_path = os.path.join(lane_config.SAVE_PATH, 'laneNet10.pth')
    model = load_model(model_path)
    data = pd.read_csv(os.path.join(os.getcwd(), "data_list", "test.csv"), header=None,
                            names=["image", "label"])
    images = data["image"].values[1:]
    for image_path in images:
        imageName = image_path.split("/")[-1]
        img = cv2.imread(image_path)
        img = img_transform(img)
        if torch.cuda.is_available():
            img = img.cuda(device=0)
        pred = model(img)
        color_mask = get_color_mask(pred)
        cv2.imwrite(os.path.join(test_dir, imageName), color_mask)
    return

if __name__ == "__main__":
    inference()
