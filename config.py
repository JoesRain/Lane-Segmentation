
class Config(object):
    # model config
    OUTPUT_STRIDE = 8
    ASPP_OUTDIM = 256
    SHORTCUT_DIM = 48
    SHORTCUT_KERNEL = 1
    NUM_CLASSES = 8

    # train config
    EPOCHS = 200
    WEIGHT_DECAY = 1.0e-4
    SAVE_PATH = "logs"
    BASE_LR = 0.0015


