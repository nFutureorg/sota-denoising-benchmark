import numpy as np
import tensorflow as tf
import os
import glob
import h5py
from skimage import io
import random
from utils import *
import scipy.io as scio
import tqdm

def make_tfrecord_SIDD(train_dir, patch_size, stride, offset=10):
    writer = tf.python_io.TFRecordWriter('./dataset/SIDD256.tfrecords')
    gt_files = []
    noisy_files = []

    batch_list = next(os.walk(train_dir))[1]
    for batch in batch_list:
        gt = glob.glob(os.path.join(train_dir, batch) + '/*GT*.PNG')
        noisy = glob.glob(os.path.join(train_dir, batch) + '/*NOISY*.PNG')
        for k in range(2):
            gt_files.append(gt[k])
            noisy_files.append(noisy[k])

    n_files = len(noisy_files)
    idx = list(range(n_files))
    random.shuffle(idx)
    for k in tqdm.tqdm(range(n_files)):
        GT_set = []
        noisy_set = []

        img = io.imread(noisy_files[idx[k]])
        h, w, c = img.shape
        if c != 3:
            img = img[:, :, :3]
        for i in range(offset, h - patch_size - offset + 1, stride):
            for j in range(offset, w - patch_size - offset + 1, stride):
                patch = img[i:i + patch_size, j:j + patch_size, :]
                noisy_set.append(patch)

        img = io.imread(gt_files[idx[k]])
        h, w, c = img.shape
        if c != 3:
            img = img[:, :, :3]
        for i in range(offset, h - patch_size - offset + 1, stride):
            for j in range(offset, w - patch_size - offset + 1, stride):
                patch = img[i:i + patch_size, j:j + patch_size, :]
                GT_set.append(patch)

        noisy_set = np.array(noisy_set)
        GT_set = np.array(GT_set)
        num_data = noisy_set.shape[0]

        for i in range(num_data):
            write_to_tfrecord(writer, noisy_set[i].tostring(), GT_set[i].tostring())
    writer.close()


def write_to_tfrecord(writer, noisy, gt):
    example = tf.train.Example(features=tf.train.Features(feature={
        'Noisy': tf.train.Feature(bytes_list=tf.train.BytesList(value=[noisy])),
        'GT': tf.train.Feature(bytes_list=tf.train.BytesList(value=[gt])),
    }))
    writer.write(example.SerializeToString())
    return


def make_realvalset(data_path):
    img = scio.loadmat(data_path + 'ValidationNoisyBlocksSrgb.mat')['ValidationNoisyBlocksSrgb']
    gt = scio.loadmat(data_path + 'ValidationGtBlocksSrgb.mat')['ValidationGtBlocksSrgb']
    (N, B, H, W, C) = img.shape
    img_ = np.reshape(img, [N * B, H, W, C])
    gt_ = np.reshape(gt, [N * B, H, W, C])
    img_ = (img_ / 255.0).astype(np.float32)
    gt_ = (gt_ / 255.0).astype(np.float32)
    np.save('./dataset/val_SIDD_noisy.npy', img_)
    np.save('./dataset/val_SIDD_gt.npy', gt_)
    return


if __name__ == '__main__':
    make_tfrecord_SIDD('./dataset/SIDD_Medium_Srgb/Data/', 256, 200)
    make_realvalset('./dataset/')
    print('Generating completed!')