import os
import argparse
from utils import *

import glob
from os.path import join
import logging
from models import *

# Parse arguments
parser = argparse.ArgumentParser(description="BoostNet Denoise")
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
parser.add_argument("--test_set", type=str, default="RNI15", help="RNI15")
parser.add_argument("--dont_save_results", action='store_true', help="don't save output images")
args = parser.parse_args()


test_set= "testsets/" + args.test_set
result_dir = "results/" + args.test_set
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# load model
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if (torch.cuda.is_available() and args.nGPU > 0) else "cpu")
if args.nGPU:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

# Create and load model
Est_rgb_realnoise = EstRGBRealNoise()
Genet_rgb = GNet()
Ganet_rgb = GNet()
Boostnet = BoostNet()
device_ids = [0]
if args.nGPU:
    state_dict_Est_rgb_realnoise = torch.load('models/EstNet_rgb_realnoise.pth')
    state_dict_Genet_rgb = torch.load('models/GENet_rgb.pth')
    state_dict_Ganet_rgb = torch.load('models/GANet_rgb.pth')
    state_dict_Boostnet = torch.load('models/BoostNet_rgb.pth')
    Est_rgb_realnoise = nn.DataParallel(Est_rgb_realnoise, device_ids=device_ids).cuda()
    Genet_rgb = nn.DataParallel(Genet_rgb, device_ids=device_ids).cuda()
    Ganet_rgb = nn.DataParallel(Ganet_rgb, device_ids=device_ids).cuda()
    Boostnet = nn.DataParallel(Boostnet, device_ids=device_ids).cuda()
else:
    # CPU mode: remove the DataParallel wrapper
    state_dict_Est_rgb_realnoise = remove_module_wrapper(
        torch.load('models/EstNet_rgb.pth', map_location=torch.device('cpu')))
    state_dict_Genet_rgb = remove_module_wrapper(
        torch.load('models/GENet_rgb.pth', map_location=torch.device('cpu')))
    state_dict_Ganet_rgb = remove_module_wrapper(
        torch.load('models/GANet_rgb.pth', map_location=torch.device('cpu')))
    state_dict_Boostnet = remove_module_wrapper(
        torch.load('models/BoostNet_rgb.pth', map_location=torch.device('cpu')))
Est_rgb_realnoise.load_state_dict(state_dict_Est_rgb_realnoise)
Genet_rgb.load_state_dict(state_dict_Genet_rgb)
Ganet_rgb.load_state_dict(state_dict_Ganet_rgb)
Boostnet.load_state_dict(state_dict_Boostnet)
Est_rgb_realnoise.eval()
Genet_rgb.eval()
Ganet_rgb.eval()
Boostnet.eval()

# Init logger
logger = logging.getLogger('testlog')
logger.setLevel(level=logging.INFO)
fh = logging.FileHandler(join(result_dir, 'log.txt'), mode='a')
formatter = logging.Formatter('%(asctime)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

print('\n> Test set')
files = []
types = ('*.bmp', '*.png', '*.jpg', '*.JPEG', '*.tif', '*.jpeg')

for tp in types:
    files.extend(glob.glob(os.path.join(test_set, tp)))
files.sort()

psnr_noi_mean, psnr_GENet_denoi_mean, psnr_GANet_denoi_mean = 0, 0, 0

for i, item in enumerate(files):
    torch.cuda.empty_cache()
    print("\tfile: %s" % item)
    img_name = os.path.basename(item)
    img_name = os.path.splitext(img_name)[0]

    # Check if input exists and if it is RGB
    try:
        rgb_den = is_rgb(item)
    except:
        raise Exception('Could not open the input image')

    if not rgb_den:
        print("Not RGB image!!!")
        continue
    imorig = preprocess(item, rgb_den)

    # Compute PSNR and log it
    logger.info("### RGB Realnoise denoising ###")
    logger.info("\tFile: " + item)
    imnoisy = imorig

    imorig = imorig.type(dtype).to(device)
    imnoisy = imnoisy.type(dtype).to(device)
    with torch.no_grad():
        if 'RNI15' in item:
            if 'Audrey_Hepburn' in img_name:
                noise_sigma = 10 / 255.
            elif 'Bears' in img_name:
                noise_sigma = 15 / 255.
            elif 'Boy' in img_name:
                noise_sigma = 45 / 255.
            elif 'Dog' in img_name:
                noise_sigma = 28 / 255.
            elif 'Flowers' in img_name:
                noise_sigma = 70 / 255.
            elif 'Frog' in img_name:
                noise_sigma = 15 / 255.
            elif 'Movie' in img_name:
                noise_sigma = 12 / 255.
            elif 'Pattern1' in img_name:
                noise_sigma = 12 / 255.
            elif 'Pattern2' in img_name:
                noise_sigma = 40 / 255.
            elif 'Pattern3' in img_name:
                noise_sigma = 25 / 255.
            elif 'Postcards' in img_name:
                noise_sigma = 15 / 255.
            elif 'Singer' in img_name:
                noise_sigma = 30 / 255.  # 30
            elif 'Stars' in img_name:
                noise_sigma = 18 / 255.
            elif 'Window' in img_name:
                noise_sigma = 15 / 255.
            elif 'Glass' in img_name:
                noise_sigma = 15 / 255.
            nsigma = Variable(torch.FloatTensor([noise_sigma]).type(dtype))
            Genet_dn = Est_rgb_realnoise(imnoisy, nsigma)
            est_noise = imnoisy - Genet_dn
            si = nsigma.view(est_noise.shape[0], 1, 1, 1).repeat(1, est_noise.shape[1], est_noise.shape[2],
                                                                 est_noise.shape[3])
            est_noi = (si, est_noise)
            Ganet_dn = Ganet_rgb(imnoisy, est_noi)
            Boostnet_dn = Boostnet(Ganet_dn, Genet_dn)
    Boostnet_dn = torch.clamp(Boostnet_dn, 0., 1.)
    # PSNR
    logger.info("\tCan not calculate PSNR in realnoise dataset")
    print("\tCan not calculate PSNR in realnoise dataset")
    # Save images
    if not args.dont_save_results:
        outimg_noi = to_cv2_image(imnoisy)
        outimg_Boost = to_cv2_image(Boostnet_dn)
        img_name_n = "{}_noisy{:02d}dB.png".format(img_name, int(noise_sigma * 255.))
        cv2.imwrite(os.path.join(result_dir, img_name_n), outimg_noi)
        img_name_noGan = '{}_BoostNet_denoi.png'.format(img_name, )
        cv2.imwrite(os.path.join(result_dir, img_name_noGan), outimg_Boost)
print('\n> Total')
print("\t#test set: {},  #total samples: {},".format(
    test_set, i + 1, ))

for handler in logger.handlers[:]:
    logger.removeHandler(handler)