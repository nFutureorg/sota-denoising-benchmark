import argparse
import json
from pprint import pprint

import matplotlib.pyplot as plt
from torchvision import transforms

import torch
import torch.nn as nn
from models import EstNetFMD, GNetFMD, BoostNetFMD
from model_n2n import N2N
from model_dncnn import DnCNN
from fmd_utils.data_loader import load_denoising_test_mix_flyv2, load_denoising_test_mix, fluore_to_tensor
from fmd_utils.metrics import cal_psnr, cal_ssim
from fmd_utils.misc import mkdirs, stitch_pathes, to_numpy
import cv2
import os

plt.switch_backend('agg')


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default='models/cell/BoostNet.pth', type=str, help='the model')
parser.add_argument('--net', type=str, default='all', choices=['N2N', 'DnCNN','EstNet', 'GeNet', 'GaNet', 'BoostNet', 'all'])
parser.add_argument('--batch-size', default=1, type=int, help='test batch size')
parser.add_argument('--data-root', default='testsets/cell', type=str, help='dir to dataset')
parser.add_argument('--out-dir', default='results/cell', type=str, help='dir to dataset')
parser.add_argument('--noise-levels', default=[1, 2, 4, 8, 16], type=str, help='dir to pre-trained model')
parser.add_argument('--image-types', type=str, default='fmd_test_mix', choices=['fmd_test_mix', 'our_data'])
parser.add_argument('--no-cuda', action='store_true', default=False, help='use GPU or not, default using GPU')
parser.add_argument('--save_img', action='store_true', default=False, help='save_img')
parser.add_argument('--cuda', type=int, default=1, help='cuda number')
opt = parser.parse_args()


test_batch_size = opt.batch_size
test_seed = 13
cmap = 'inferno'
device = 'cpu' if opt.no_cuda else 'cuda'

noise_levels = opt.noise_levels

if opt.image_types == 'all':
    image_types = ['fmd_test_mix', 'our_data']
else:
    image_types = [opt.image_types]

if opt.net == 'all':
    nets = ['N2N', 'DnCNN','EstNet', 'GeNet', 'GaNet', 'BoostNet', 'all']
else:
    nets = [opt.net]

for image_type in image_types:
    for net in nets:
        print('#### NET : %s ####'%net)
        if net == 'N2N':
            model = N2N(1, 1).to(device)
            model.load_state_dict(torch.load('models/cell/n2n.pth'))
        elif net == 'DnCNN':
            model = DnCNN(depth=17,
                          n_channels=64,
                          image_channels=1,
                          use_bnorm=True,
                          kernel_size=3).to(device)
            model.load_state_dict(torch.load('models/cell/dncnn.pth'))
        elif net == 'EstNet':
            model = EstNetFMD().to(device)
            model = nn.DataParallel(model, list(range(opt.cuda)))
            model.load_state_dict(torch.load('models/cell/EstNet_%s.pth'%image_type))
        elif net == 'GeNet':
            model0 = EstNetFMD().to(device)
            model0 = nn.DataParallel(model0, list(range(opt.cuda)))
            model0.load_state_dict(torch.load('models/cell/EstNet_%s.pth'%image_type))
            model = GNetFMD().to(device)
            model = nn.DataParallel(model, list(range(opt.cuda)))
            model.load_state_dict(torch.load('models/cell/GENet_%s.pth'%image_type))
            model0.eval()
        elif net == 'GaNet':
            model0 = EstNetFMD().to(device)
            model0 = nn.DataParallel(model0, list(range(opt.cuda)))
            model0.load_state_dict(torch.load('models/cell/EstNet_%s.pth'%image_type))
            model = GNetFMD().to(device)
            model = nn.DataParallel(model, list(range(opt.cuda)))
            model.load_state_dict(torch.load('models/cell/GANet_%s.pth'%image_type))
        elif net == 'BoostNet':
            model0 = EstNetFMD().to(device)
            model1 = GNetFMD().to(device)
            model2 = GNetFMD().to(device)
            model0 = nn.DataParallel(model0, list(range(opt.cuda)))
            model1 = nn.DataParallel(model1, list(range(opt.cuda)))
            model2 = nn.DataParallel(model2, list(range(opt.cuda)))
            model0.load_state_dict(torch.load('models/cell/EstNet_%s.pth'%image_type))
            model1.load_state_dict(torch.load('models/cell/GENet_%s.pth'%image_type))
            model2.load_state_dict(torch.load('models/cell/GANet_%s.pth'%image_type))

            model = BoostNetFMD().to(device)

            model = nn.DataParallel(model, list(range(opt.cuda)))
            model.load_state_dict(torch.load('models/cell/BoostNet_%s.pth'%image_type))


        def convert(x):
            x = (x.transpose(1,2,0)+0.5)*255
            return x.clip(0,255).astype('uint8')


        logger = {}
        four_crop = transforms.Compose([
            transforms.FiveCrop(256),
            transforms.Lambda(lambda crops: torch.stack([
                fluore_to_tensor(crop) for crop in crops[:4]])),
            transforms.Lambda(lambda x: x.float().div(255).sub(0.5))
            ])

        for noise_level in noise_levels:
            out_dir = os.path.join(opt.out_dir, image_type, net)
            mkdirs(out_dir)
            if noise_level==1:
                test_case_dir = out_dir + '/raw/'
            else:
                test_case_dir = out_dir + f'/avg{noise_level}/'
            mkdirs(test_case_dir)
            if image_type == 'fmd_test_mix':
                n_plots = 12
                test_loader = load_denoising_test_mix(opt.data_root,
                                                      batch_size=test_batch_size, noise_levels=[noise_level],
                                                      transform=four_crop, target_transform=four_crop,
                                                      patch_size=256)
            elif image_type == 'our_data':
                n_plots = 12
                test_loader = load_denoising_test_mix_flyv2(opt.data_root,
                                                            batch_size=test_batch_size, noise_levels=[noise_level],
                                                            transform=four_crop, target_transform=four_crop,
                                                            patch_size=256)
            # four crop
            multiplier = 4
            n_test_samples = len(test_loader.dataset) * multiplier

            case = {'noise': noise_level,
                    'type': image_type,
                    'samples': n_test_samples,
                    }
            pprint(case)
            print('Start testing............')

            psnr, ssim = 0., 0.
            psnr_noi, ssim_noi = 0., 0.
            out = {}
            for batch_idx, (noisy, clean, path, noilv_input) in enumerate(test_loader):
                name = os.path.basename(path[0])
                noisy, clean, noilv_input = noisy.to(device), clean.to(device), noilv_input.to(device)
                # fuse batch and four crop
                noisy = noisy.view(-1, *noisy.shape[2:])
                clean = clean.view(-1, *clean.shape[2:])
                noilv_input = noilv_input.view(-1, *noilv_input.shape[2:])
                with torch.no_grad():
                    if net == 'EstNet':
                        denoised = model(noisy, noilv_input)
                    elif net == 'GaNet' or net == 'GeNet':
                        denoised0 = model0(noisy, noilv_input)
                        est_noise = noisy - denoised0
                        si = noilv_input.view(est_noise.shape[0], 1, 1, 1).repeat(1, est_noise.shape[1],
                                                                                  est_noise.shape[2],
                                                                                  est_noise.shape[3])
                        est_noi = (si, est_noise)
                        denoised = model(noisy, est_noi)
                    elif net == 'BoostNet':
                        denoised0 = model0(noisy, noilv_input)
                        est_noise = noisy - denoised0
                        si = noilv_input.view(est_noise.shape[0], 1, 1, 1).repeat(1, est_noise.shape[1],
                                                                                  est_noise.shape[2],
                                                                                  est_noise.shape[3])
                        est_noi = (si, est_noise)
                        denoised1 = model1(noisy, est_noi)
                        denoised2 = model2(noisy, est_noi)
                        denoised = model(noisy,denoised1, denoised2)
                    else:
                        denoised = model(noisy)

                psnr += cal_psnr(clean, denoised).sum().item()
                ssim += cal_ssim(clean, denoised).sum()

                psnr_noi += cal_psnr(clean, noisy).sum().item()
                ssim_noi += cal_ssim(clean, noisy).sum()

            psnr = psnr / n_test_samples
            ssim = ssim / n_test_samples

            psnr_noi = psnr_noi / n_test_samples
            ssim_noi = ssim_noi / n_test_samples
            print('Noisy PSNR/SSIM: %.2f/%.4f'%(psnr_noi,ssim_noi))

            result = {'psnr_dn': '%.2f'%psnr,
                      'ssim_dn': '%.4f'%ssim,}
            case.update(result)
            pprint(result)
            logger.update({f'noise{noise_level}_{image_type}': case})

            with open(out_dir + "/results_{}_{}.txt".format('cpu' if opt.no_cuda else 'gpu', image_type), 'w') as args_file:
                json.dump(logger, args_file, indent=4)

            if opt.save_img:
                fixed_denoised_stitched = stitch_pathes(to_numpy(denoised))
                out[test_case_dir + name.replace('.png', '_dn.png')] = convert(fixed_denoised_stitched)
                cv2.imwrite(test_case_dir + name.replace('.png', '_dn.png'), convert(fixed_denoised_stitched))



