import time, itertools
from model.dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from model.networks import *
from model.utils import *
from glob import glob

class UGATIT(object) :
    def __init__(self, args):
        # To load light model
        self.light = args.light
        if self.light :
            self.model_name = 'UGATIT_light'
        else :
            self.model_name = 'UGATIT'
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        """ Generator """
        self.n_res = args.n_res
        self.ch = args.ch
        self.img_size = args.img_size
        self.img_ch = args.img_ch
        # Device
        self.device = args.device

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ Define Generator, Discriminator """
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)

        """ Define Loss """
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)
        """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
        # self.Rho_clipper = RhoClipper(0, 1)

    def load(self, dir, step):
        params = torch.load(os.path.join(dir, self.dataset + '_params_%07d.pt' % step), map_location=torch.device('cpu'))
        self.genA2B.load_state_dict(params['genA2B'])

    def test(self, img):
        # Check if model exists
        model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
        if not len(model_list) == 0:
            model_list.sort()
            iter = int(model_list[-1].split('_')[-1].split('.')[0])
            # Load latest model
            self.load(os.path.join(self.result_dir, self.dataset, 'model'), iter)
        else:
            return 'error'
        # Set model to evaluation mode (production, so weights freeze)
        self.genA2B.eval()
        # Transform image to fit model best
        img_size = 256
        test_transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ])
        # Convert image to tensor
        img_tensor = test_transform(img).unsqueeze_(0)
        
        # Load image in device (memory)
        img = img_tensor.to(self.device)
        # Forward pass
        fake_A2B, _, fake_A2B_heatmap = self.genA2B(img)
        # Convert to RGB and return
        out = RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))*255
        out = out.astype(int)
        return out