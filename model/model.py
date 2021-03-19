from model.UGATIT import UGATIT
from model.utils import *
from io import BytesIO
import requests
from PIL import Image
from torchvision import transforms

class args_loader:
    def __init__(self):
        self.phase = 'test'
        self.light = True
        self.device = 'cpu'
        self.n_res = 4
        self.img_size = 200
        self.img_ch = 3
        self.ch = 64
        self.result_dir = 'model/results'
        self.dataset = 'testing'

"""main"""
def cartoonify(img):
    # response = requests.get(url)
    # img = Image.open(BytesIO(response.content)).convert('RGB')
    # open_cv_image = np.array(img) 
    # img = open_cv_image[:, :, ::-1].copy() 
    # parse arguments
    args = args_loader()
    # open session
    gan = UGATIT(args)
    # build graph
    gan.build_model()
    # Image to tensor to be accepted by model

    # Forward pass
    img2 = gan.test(img)
    return img2
