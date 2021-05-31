import os
import numpy as np
import torch
import cv2
import mrc
from libtiff import TIFF
from medpy import metric
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
SMOOTH = 1e-6

'''
# --------------------------------------------
# https://github.com/twhui/SRGAN-pyTorch
# https://github.com/xinntao/BasicSR
# https://github.com/pytorch/pytorch/issues/1249
# https://github.com/cszn
# https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
# --------------------------------------------
'''
NPY_EXTENSIONS = ['.npy', '.NPY']

# --------------------------------------------
# get image pathes
# --------------------------------------------
def get_npys_paths(dataroot):
    paths = None  # return None if dataroot is None
    if dataroot is not None:
        paths = sorted(_get_paths_from_npys(dataroot))
    return paths

# --------------------------------------------
# input: path saving npy files
# output: list of npy files
# --------------------------------------------
def _get_paths_from_npys(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_npy_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images

# --------------------------------------------
# to judge the input file is a .npy file or not
# --------------------------------------------
def is_npy_file(filename):
    return any(filename.endswith(extension) for extension in NPY_EXTENSIONS)



# --------------------------------------------
# matlab-fashion mkdir
# --------------------------------------------
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)



# --------------------------------------------
# read image from path
# --------------------------------------------
def imread_training(path, n_channels=1):
    assert path[-3:] == 'npy', 'only .npy data are allowed while training'
    img_np = np.load(path)
    assert np.shape(img_np)[2] == n_channels, 'error in assert np.shape(input_npy_data)[0] == n_channels'
    return img_np

def labelread_training(path, n_channels=1):
    assert path[-3:] == 'npy', 'only .npy data are allowed while training'
    label_np = np.load(path)
    assert label_np.ndim == 2, 'error in assert label_np.ndim == 2'
    return label_np



# --------------------------------------------
# matlab-fashion imwrite using opencv lib
# --------------------------------------------
# It's said opencv is faster than scipy.io (now, imageio)
# --------------------------------------------
def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)

def imwrite(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)



# --------------------------------------------
# image format conversion
# numpy (uint8)   <--->  tensor
# --------------------------------------------
def uint82tensor(img):
    img = img.astype(np.float32)
    img = img / 255.0
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()

def label_uint82tensor(img):
    return torch.from_numpy(np.ascontiguousarray(img.astype(np.int16))).to(torch.long)

def tensor2uint8(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

def label_tensor2uint8(img):
    img = img.data.squeeze().cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8(img)


# --------------------------------------------
# Augmentation, flipe and/or rotate
# --------------------------------------------
# augmet_img: numpy image of WxHxC or WxH
# It can also in fact used for numpy image of HxWxC or HxW
# --------------------------------------------
def augment_img(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))



# --------------------------------------------
# Read Mrc Image and convert to ImageJ-type numpy matrix
# --------------------------------------------
def ReadMrcImage(path):
    f = mrc.imread(path)
    f = np.array(f)
    f = f[..., -1::-1, :]  # [b,c,-h,w] - > [b,c,h,w]
    return f

# --------------------------------------------
# Save numpy matrix to MRC files
# --------------------------------------------
def SaveMrcImage(f, path):
    f = f[..., -1::-1, :] # [b,c,h,w] - > [b,c,-h,w]
    mrc.save(f, path)

# --------------------------------------------
# Read tif files
# --------------------------------------------
def tiff2Stack(filePath):
    tif = TIFF.open(filePath,mode='r')
    stack = []
    for img in list(tif.iter_images()):
        stack.append(img)
    return np.array(stack)

# --------------------------------------------
# Save numpy matrix to tif files
# --------------------------------------------
def Stack2tiff(img, filePath):
    tif = TIFF.open(filePath,mode='w')
    for i in range(0, img.shape[0]):
        tif.write_image(img[i,:,:], compression=None, write_rgb=True)
    tif.close()
    return


# --------------------------------------------
# Segmentation evaluation index
# --------------------------------------------
def caL_dice(pred, label):
    pred = pred.cpu().numpy()
    label = label.cpu().numpy()
    pred[pred > 0] = 1
    label[label > 0] = 1
    if pred.sum() > 0 and label.sum()>0:
        return metric.binary.dc(pred, label)
    elif pred.sum() > 0 and label.sum()==0:
        return 1
    else:
        return 0

def cal_iou(outputs: torch.Tensor, labels: torch.Tensor):
    # input: BATCH x H x W shape
    # outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    if len(outputs.shape) == 2: outputs = outputs.unsqueeze(0)
    if len(labels.shape) == 2: labels = labels.unsqueeze(0)
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    return thresholded.mean().squeeze().cpu().numpy()  # Or thresholded.mean() if you are interested in average across the batch