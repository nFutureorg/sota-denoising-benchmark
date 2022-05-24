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
parser.add_argument("--test_set", type=str, default="Set5", help="Set5, Set15, Kodak24, LIVE1, McMaster or CBSD68")
parser.add_argument('--noise_sigma', type=int, default=25, help='noisy sigma')
parser.add_argument("--dont_save_results", action='store_true', help="don't save output images")
args = parser.parse_args()


test_set= "testsets/" + args.test_set
result_dir = "results/" + args.test_set + '_' + str(args.noise_sigma)
args.noise_sigma = args.noise_sigma/255.
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
Estnet_rgb = EstNetRGB()
Genet_rgb = GNet()
Ganet_rgb = GNet()
Boostnet = BoostNet()
device_ids = [0]
if args.nGPU:
    state_dict_Estnet_rgb = torch.load('models/EstNet_rgb.pth')
    state_dict_Genet_rgb = torch.load('models/GENet_rgb.pth')
    state_dict_Ganet_rgb = torch.load('models/GANet_rgb.pth')
    state_dict_Boostnet = torch.load('models/BoostNet_rgb.pth')
    Estnet_rgb = nn.DataParallel(Estnet_rgb, device_ids=device_ids).cuda()
    Genet_rgb = nn.DataParallel(Genet_rgb, device_ids=device_ids).cuda()
    Ganet_rgb = nn.DataParallel(Ganet_rgb, device_ids=device_ids).cuda()
    Boostnet = nn.DataParallel(Boostnet, device_ids=device_ids).cuda()
else:
    # CPU mode: remove the DataParallel wrapper
    state_dict_Estnet_rgb = remove_module_wrapper(
        torch.load('models/EstNet_rgb.pth', map_location=torch.device('cpu')))
    state_dict_Genet_rgb = remove_module_wrapper(
        torch.load('models/GENet_rgb.pth', map_location=torch.device('cpu')))
    state_dict_Ganet_rgb = remove_module_wrapper(
        torch.load('models/GANet_rgb.pth', map_location=torch.device('cpu')))
    state_dict_Boostnet = remove_module_wrapper(
        torch.load('models/BoostNet_rgb.pth', map_location=torch.device('cpu')))
Estnet_rgb.load_state_dict(state_dict_Estnet_rgb)
Genet_rgb.load_state_dict(state_dict_Genet_rgb)
Ganet_rgb.load_state_dict(state_dict_Ganet_rgb)
Boostnet.load_state_dict(state_dict_Boostnet)
Estnet_rgb.eval()
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

psnr_noi_mean, psnr_GENet_denoi_mean, psnr_GANet_denoi_mean, psnr_BoostNet_denoi_mean = 0, 0, 0, 0

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
    logger.info("### RGB Add-noise denoising ###")
    logger.info("\tFile: " + item)

    noise = torch.FloatTensor(imorig.size()).normal_(mean=0, std=args.noise_sigma)
    imnoisy = imorig + noise

    imorig = imorig.type(dtype).to(device)
    imnoisy = imnoisy.type(dtype).to(device)
    noise_sigma = torch.FloatTensor([args.noise_sigma]).type(dtype).to(device)
    with torch.no_grad():
        est_noi = Estnet_rgb(imnoisy)
        Genet_dn = Genet_rgb(imnoisy, est_noi)
        Ganet_dn = Ganet_rgb(imnoisy, est_noi)
        Boostnet_dn = Boostnet(Genet_dn, Ganet_dn)
    Boostnet_dn = torch.clamp(Boostnet_dn, 0., 1.)
    # PSNR
    psnr_noisy = batch_psnr(imnoisy, imorig, data_range=1.)
    psnr_noi_mean += psnr_noisy
    psnr_BoostNet_denoi = batch_psnr(Boostnet_dn, imorig, data_range=1.)
    psnr_BoostNet_denoi_mean += psnr_BoostNet_denoi
    logger.info("\tPSNR denoised (noi/BoostNet_denoi):  {:0.2f}/{:0.2f}dB".
                format(psnr_noisy, psnr_BoostNet_denoi))
    print("\tPSNR denoised (noi/BoostNet_denoi):  {:0.2f}/{:0.2f}dB".
                format(psnr_noisy, psnr_BoostNet_denoi))
    # Save images
    if not args.dont_save_results:
        outimg_noi = to_cv2_image(imnoisy)
        outimg_Boost_dn = to_cv2_image(Boostnet_dn)
        img_name_n = "{}_noisy{:02d}_{:0.2f}dB.png".format(img_name, int(args.noise_sigma * 255.),
                                                           psnr_noisy)
        cv2.imwrite(os.path.join(result_dir, img_name_n), outimg_noi)
        img_name_ = '{}_BoostNet_denoi{:0.2f}.png'.format(img_name, psnr_BoostNet_denoi)
        cv2.imwrite(os.path.join(result_dir, img_name_), outimg_Boost_dn)

psnr_noi_mean /= (i + 1)
psnr_GENet_denoi_mean /= (i + 1)
psnr_GANet_denoi_mean /= (i + 1)
psnr_BoostNet_denoi_mean /=(i + 1)
logger.info("\n\t#test set: {}, : Sigma: {}   #total samples: {},".format(
    test_set, int(args.noise_sigma * 255.), i + 1, ))
logger.info("\t\t#mean psnr (noi/BoostNet):  {:0.2f}/{:0.2f}dB".format(
    psnr_noi_mean, psnr_BoostNet_denoi_mean))
print('\n> Total')
print("\t#test set: {}, : Sigma: {}   #total samples: {},".format(
    test_set, int(args.noise_sigma * 255.), i + 1, ))
print("\t\t#mean psnr (noi/BoostNet):  {:0.2f}/{:0.2f}dB".format(
    psnr_noi_mean, psnr_BoostNet_denoi_mean))

for handler in logger.handlers[:]:
    logger.removeHandler(handler)