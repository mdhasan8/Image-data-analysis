# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 14:27:06 2020

@author: Easin
"""

import torch
from efficientnet_pytorch import model as enet
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from tqdm.notebook import tqdm
import skimage.io
from sklearn.metrics import cohen_kappa_score
import numpy as np
import cv2
import math
import warnings

enet_type = 'efficientnet-b2'
out_dim = 5
COLS = 6
ROWS = 6
n_tiles = COLS*ROWS
image_size = 256
batch_size = 2
num_workers = 4
tile_size = 256
SIZE = tile_size
N = n_tiles
LAYER = 1 # medium
WINDOW_SIZE = 256
STRIDE = 256
K = 36

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = '../input/prostate-cancer-grade-assessment'

#set this as 'train' to test on train data. set it as 'submit' for submission
train_or_submit = 'submit'  

if train_or_submit=='train':
    df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    df = df.head(5)
    df_K = df[df.data_provider=='karolinska'].reset_index()
    df_R = df[df.data_provider=='radboud'].reset_index()
    image_folder = os.path.join(data_dir, 'train_images')
else:
    df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    df_K = df[df.data_provider=='karolinska'].reset_index()
    df_R = df[df.data_provider=='radboud'].reset_index()    
    image_folder = os.path.join(data_dir, 'test_images')
    
class enetv2(nn.Module):
    def __init__(self, backbone, out_dim):
        super(enetv2, self).__init__()
        self.enet = enet.EfficientNet.from_name(backbone)

        self.myfc = nn.Linear(self.enet._fc.in_features, out_dim)
        self.enet._fc = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x
    
def white_img_flag(img):
    #if all white isup grade is 0, for else it will predict
    h, w, c = img.shape
    if img.sum() == h * w * c * 255:
        return 'Y'
    else:
        return 'N'
    
def get_tiles(img, mode=0):
        result = []
        h, w, c = img.shape
        pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
        pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)

        img2 = np.pad(img,[[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2,pad_w - pad_w//2], [0,0]], constant_values=255)
        img3 = img2.reshape(
            img2.shape[0] // tile_size,
            tile_size,
            img2.shape[1] // tile_size,
            tile_size,
            3
        )

        img3 = img3.transpose(0,2,1,3,4).reshape(-1, tile_size, tile_size,3)
        n_tiles_with_info = (img3.reshape(img3.shape[0],-1).sum(1) < tile_size ** 2 * 3 * 255).sum()
        if len(img3) < n_tiles:
            img3 = np.pad(img3,[[0,n_tiles-len(img3)],[0,0],[0,0],[0,0]], constant_values=255)
        idxs = np.argsort(img3.reshape(img3.shape[0],-1).sum(-1))[:n_tiles]
        img3 = img3[idxs]
#         print(img3.shape, len(img3))
        for i in range(len(img3)):
            result.append({'img':img3[i], 'idx':i})
        return result, n_tiles_with_info >= n_tiles

def akensert_tiles(img:np.ndarray, debug=False)->np.ndarray:    
    
    # get tile coords
    img, coords = compute_coords(
        img,
        patch_size=SIZE,
        precompute=False, # returns new padded img
        min_patch_info=0.35,
        min_axis_info=0.35,
        min_consec_axis_info=0.35,
        min_decimal_keep=0.7)
    
    # sort coords (high info -> low info)
    coords = sorted(coords, key= lambda x: x[0], reverse=False)
    
    # select top N tiles
    tiles = []
    for i in range(len(coords)):
        if i == N:
            break;
        _, x, y = coords[i]
        tiles.append(img[x:x+SIZE,y:y+SIZE])
    
    # append white tiles if necessary
    selected = np.array(tiles)
    if len(selected)<N:
        selected = np.pad(
            selected,
            [[0,N-len(selected)],[0,0],[0,0],[0,0]],
            constant_values=255
        )
    
    # merge tiles to one image
    merged = join_tiles(selected)
    
    if debug:
        for (v, y, x) in coords:
            img = cv2.rectangle(img, (x, y), (x+SIZE, y+SIZE), color=(0, 0, 0), thickness=5)
            img = cv2.circle(img, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
            img = cv2.circle(img, (x+SIZE, y+SIZE), radius=5, color=(0, 255, 0), thickness=-1)
        return merged, img
    else:
        return merged

def join_tiles(img:np.ndarray)->np.ndarray:
    reshaped = img.reshape(
        COLS,
        ROWS,    
        img.shape[1],
        img.shape[2],
        3
    )
    transposed = reshaped.transpose(0, 2, 1, 3, 4)
    return transposed.reshape(COLS * SIZE, ROWS * SIZE, 3)

def enhance_image(image, contrast=1, brightness=15):
    """
    Enhance constrast and brightness of images
    """
    img_enhanced = cv2.addWeighted(image, contrast, image, 0, brightness)
    return img_enhanced

def unsharp_masking(img):
    """ Unsharp masking of an RGB image"""
    img_gaussian = cv2.GaussianBlur(img, (21,21), 10.0)
    return cv2.addWeighted(img, 1.8, img_gaussian, -0.8, 0, img)

def _mask_tissue(image, kernel_size=(7, 7), gray_threshold=220):
    """Masks tissue in image. Uses gray-scaled image, as well as
    dilation kernels and 'gap filling'
    """
    # Define elliptic kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    # Convert rgb to gray scale for easier masking
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Now mask the gray-scaled image (capturing tissue in biopsy)
    mask = np.where(gray < gray_threshold, 1, 0).astype(np.uint8)
    # Use dilation and findContours to fill in gaps/holes in masked tissue
    mask = cv2.dilate(mask, kernel, iterations=1)
    contour, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(mask, [cnt], 0, 1, -1)
    return mask

def _pad_image(image, pad_len, pad_val):
    """Pads inputted image, accepts both 
    2-d (mask) and 3-d (rgb image) arrays
    """
    if image is None:
        return None
    elif image.ndim == 2:
        return np.pad(
            image, ((pad_len, pad_len), (pad_len, pad_len)), pad_val)
    elif image.ndim == 3:
        return np.pad(
            image, ((pad_len, pad_len), (pad_len, pad_len), (0, 0)), pad_val)
    return None

def _get_tissue_parts_indices(tissue, min_consec_info):
    """If there are multiple tissue parts in 'tissue', 'tissue' will be 
    split. Each tissue part will be taken care of separately (later on), 
    and if the tissue part is less than min_consec_info, it's considered 
    to small and won't be returned.
    """
    split_points = np.where(np.diff(tissue) != 1)[0]+1
    tissue_parts = np.split(tissue, split_points)
    return [
        tp for tp in tissue_parts if len(tp) >= min_consec_info
    ]

def _transpose_image(image):
    """Inputs an image and transposes it, accepts 
    both 2-d (mask) and 3-d (rgb image) arrays
    """
    if image is None:
        return None
    elif image.ndim == 2:
        return np.transpose(image, (1, 0)).copy()
    elif image.ndim == 3:
        return np.transpose(image, (1, 0, 2)).copy()
    return None

def _get_tissue_subparts_coords(subtissue, patch_size, min_decimal_keep):
    """Inputs a tissue part resulting from '_get_tissue_parts_indices'.
    This tissue part is divided into N subparts and returned.
    Argument min_decimal_keep basically decides if we should reduce the
    N subparts to N-1 subparts, due to overflow.
    """
    start, end = subtissue[0], subtissue[-1]
    num_subparts = (end-start)/patch_size
    if num_subparts % 1 < min_decimal_keep and num_subparts >= 1:
        num_subparts = math.floor(num_subparts)
    else:
        num_subparts = math.ceil(num_subparts)

    excess = (num_subparts*patch_size) - (end-start)
    shift = excess // 2

    return [
        i * patch_size + start - shift 
        for i in range(num_subparts)
    ]

def _eval_and_append_xy_coords(coords,
                               image, 
                               mask, 
                               patch_size, 
                               x, y, 
                               min_patch_info,
                               transposed,
                               precompute):
    """Based on computed x and y coordinates of patch: 
    slices out patch from original image, flattens it,
    preprocesses it, and finally evaluates its mask.
    If patch contains more info than min_patch_info,
    the patch coordinates are kept, along with a value 
    'val1' that estimates how much information there 
    is in the patch. Smaller 'val1' assumes more info.
    """
    patch_1d = (
        image[y: y+patch_size, x:x+patch_size, :]
        .mean(axis=2)
        .reshape(-1)
    )
    idx_tissue = np.where(patch_1d <= 210)[0]
    idx_black = np.where(patch_1d < 5)[0]
    idx_background = np.where(patch_1d > 210)[0]

    if len(idx_tissue) > 0:
        patch_1d[idx_black] = 210
        patch_1d[idx_background] = 210
        val1 = int(patch_1d.mean())
        val2 = mask[y:y+patch_size, x:x+patch_size].mean()
        if val2 > min_patch_info:
            if precompute:
                if transposed:
                    coords = np.concatenate([
                        coords, [[val1, x-patch_size, y-patch_size]]
                    ])
                else:
                    coords = np.concatenate([
                        coords, [[val1, y-patch_size, x-patch_size]]
                    ])
            else:
                coords = np.concatenate([
                    coords, [[val1, y, x]]
                ])
               
    return coords

def compute_coords(image,
                   patch_size=256,
                   precompute=False,
                   min_patch_info=0.35,
                   min_axis_info=0.35,
                   min_consec_axis_info=0.35,
                   min_decimal_keep=0.7):

    """
    Input:
        image : 3-d np.ndarray
        patch_size : size of patches/tiles, will be of 
            size (patch_size x patch_size x 3)
        precompute : If True, only coordinates will be returned,
            these coordinates match the inputted 'original' image.
            If False, both an image and coordinates will be returned,
            the coordinates does not match the inputted image but the
            image that it is returned with.
        min_patch_info : Minimum required information in patch
            (see '_eval_and_append_xy_coords')
        min_axis_info : Minimum fraction of on-bits in x/y dimension to be 
            considered enough information. For x, this would be fraction of 
            on-bits in x-dimension of a y:y+patch_size slice. For y, this would 
            be the fraction of on-bits for the whole image in y-dimension
        min_consec_axis_info : Minimum consecutive x/y on-bits
            (see '_get_tissue_parts_indices')
        min_decimal_keep : Threshold for decimal point for removing "excessive" patch
            (see '_get_tissue_subparts_coords')
    
    Output:
        image [only if precompute is False] : similar to input image, but fits 
            to the computed coordinates
        coords : the coordinates that will be used to compute the patches later on
    """
    
    
    if type(image) != np.ndarray:
        # if image is a Tensor
        image = image.numpy()
    
    # masked tissue will be used to compute the coordinates
    mask = _mask_tissue(image)

    # initialize coordinate accumulator
    coords = np.zeros([0, 3], dtype=int)

    # pad image and mask to make sure no tissue is potentially missed out
    image = _pad_image(image, patch_size, 'maximum')
    mask = _pad_image(mask, patch_size, 'minimum')
    
    y_sum = mask.sum(axis=1)
    x_sum = mask.sum(axis=0)
    # if on bits in x_sum is greater than in y_sum, the tissue is
    # likely aligned horizontally. The algorithm works better if
    # the image is aligned vertically, thus the image will be transposed
    if len(np.where(x_sum > 0)[0]) > len(np.where(y_sum > 0)[0]):
        image = _transpose_image(image)
        mask = _transpose_image(mask)
        y_sum, _ = x_sum, y_sum
        transposed = True
    else:
        transposed = False
    
    # where y_sum is more than the minimum number of on-bits
    y_tissue = np.where(y_sum >= (patch_size*min_axis_info))[0]
    
    if len(y_tissue) < 1:
        warnings.warn("Not enough tissue in image (y-dim)", RuntimeWarning)
        if precompute: return [(0, 0, 0)]
        else: return image, [(0, 0, 0)]
    
    y_tissue_parts_indices = _get_tissue_parts_indices(
        y_tissue, patch_size*min_consec_axis_info)
    
    if len(y_tissue_parts_indices) < 1: 
        warnings.warn("Not enough tissue in image (y-dim)", RuntimeWarning)
        if precompute: return [(0, 0, 0)]
        else: return image, [(0, 0, 0)]
    
    # loop over the tissues in y-dimension
    for yidx in y_tissue_parts_indices:
        y_tissue_subparts_coords = _get_tissue_subparts_coords(
            yidx, patch_size, min_decimal_keep)
        
        for y in y_tissue_subparts_coords:
            # in y_slice, where x_slice_sum is more than the minimum number of on-bits
            x_slice_sum = mask[y:y+patch_size, :].sum(axis=0)
            x_tissue = np.where(x_slice_sum >= (patch_size*min_axis_info))[0]
            
            x_tissue_parts_indices = _get_tissue_parts_indices(
                x_tissue, patch_size*min_consec_axis_info)
            
            # loop over tissues in x-dimension (inside y_slice 'y:y+patch_size')
            for xidx in x_tissue_parts_indices:
                x_tissue_subparts_coords = _get_tissue_subparts_coords(
                    xidx, patch_size, min_decimal_keep)
                
                for x in x_tissue_subparts_coords:
                    coords = _eval_and_append_xy_coords(
                        coords, image, mask, patch_size, x, y, 
                        min_patch_info, transposed, precompute
                    )     
    
    if len(coords) < 1:
        warnings.warn("Not enough tissue in image (x-dim)", RuntimeWarning)
        if precompute: return [(0, 0, 0)]
        else: return image, [(0, 0, 0)]
    
    if precompute: return coords
    else: return image, coords
    
def get_row_dict(image , num_rows , window_size, patches ):
    
    total_pixels = image.shape[0] * image.shape[1]
    total_white = compute_statistics_lite(image)
    total_nonwhite = total_pixels - total_white
    
    row_dict = {key: 0 for key in range(num_rows)}
    width = image.shape[1]
    
    for i in range(num_rows):
        white = compute_statistics_lite(image[i*window_size:(i+1)*window_size, :, :])
        nonwhite = (window_size*width)- white 
        row_dict[i] = (patches * nonwhite)//total_nonwhite 
    
    row_num = 0 
    while (patches - sum(row_dict.values())) > 0 :
        row_dict[row_num] = row_dict[row_num] + 1
        row_num = (row_num + 1) % num_rows
    
#    print(row_dict)
    return row_dict

def generate_patches(image, window_size=200, stride=128, k=20):
    
    image = np.array(image)    
    max_height, max_width  = image.shape[0], image.shape[1]   
    num_rows = max_height//window_size
    patches = k 
    row_dict = get_row_dict(image , num_rows , window_size, patches) # gets the patches needed from each rows  
    regions_container = []
    i = 0
    
    while window_size + window_size*i <= max_height:
        j = 0
        row_container = []                     
        while window_size + stride*j <= max_width:            
            x_top_left_pixel = i * window_size
            y_top_left_pixel = j * stride
            
            patch = image[
                x_top_left_pixel : x_top_left_pixel + window_size,
                y_top_left_pixel : y_top_left_pixel + window_size,
                :
            ]
            
            ratio_white_pixels, green_concentration, blue_concentration = compute_statistics(patch)
            
            
            region_tuple = (x_top_left_pixel, y_top_left_pixel, ratio_white_pixels, green_concentration, blue_concentration)
            if ratio_white_pixels <= .9:
                row_container.append(region_tuple)
            
            j += 1
        row_container = sorted(row_container, key = lambda x: x[2])
        
        #print(len(row_container) ,  row_dict[i] )
        if len(row_container) < row_dict[i]:
            for m in range(row_dict[i] - len(row_container)):
                #print(m)
                #print(row_container[m])
                row_container.append(row_container[m])
                
        regions_container.extend(row_container[:row_dict[i]]) 
        i += 1

# as we are already selecting the k patches we dont need the select function
    k_best_region_coordinates = regions_container ### select_k_best_regions(regions_container, k=k)
# putting the region container directly into the get k best regions
    k_best_regions = get_k_best_regions(k_best_region_coordinates, image, window_size)
#     Show_patchings(image, k_best_region_coordinates, WINDOW_SIZE)
    
    return image, k_best_region_coordinates, k_best_regions

def get_patch_image(image, window_size=WINDOW_SIZE, stride=STRIDE, k=K):
#     print('get_patch_image', image.shape)
    image, best_coordinates, best_regions = generate_patches(image, window_size, stride, k)
    glued_image = glue_to_one_picture(best_regions, window_size, k)
    return glued_image

def compute_statistics(image):
    """
    Args:
        image                  numpy.array   multi-dimensional array of the form WxHxC
    
    Returns:
        ratio_white_pixels     float         ratio of white pixels over total pixels in the image 
    """
    height , width = image.shape[0], image.shape[1]

    num_pixels = width * height
    if num_pixels == 0:
        return 1,0,0
    
    
    num_white_pixels = 0
    
    summed_matrix = np.sum(image, axis=-1)> 620
    truth_matrix = np.multiply(abs(image[:,:,0]-image[:,:,1])<=20,abs(image[:,:,1]-image[:,:,2])<=20)
    white_pixels = np.multiply(summed_matrix , truth_matrix)
    num_white_pixels = np.count_nonzero(white_pixels)
    
    
    # Note: A 3-channel white pixel has RGB (255, 255, 255)
    #num_white_pixels = np.count_nonzero(summed_matrix > 620)
    ratio_white_pixels = num_white_pixels / num_pixels
    
    green_concentration = np.mean(image[1])
    blue_concentration = np.mean(image[2])
    
    return ratio_white_pixels, green_concentration, blue_concentration

def compute_statistics_lite(image):
    #print(image.shape)
    height , width = image.shape[0], image.shape[1]
    num_pixels = width * height
    if num_pixels == 0:
        return 1
    
    num_white_pixels = 0
    summed_matrix = np.sum(image, axis=-1)> 620
    truth_matrix = np.multiply(abs(image[:,:,0]-image[:,:,1])<=20,abs(image[:,:,1]-image[:,:,2])<=20)
    white_pixels = np.multiply(summed_matrix , truth_matrix)
    num_white_pixels = np.count_nonzero(white_pixels) 
    return num_white_pixels

def padding_image(image):
    image = np.array(image)    
    max_width, max_height , channels  = image.shape
#    print( max_width, max_height , channels)
    final_width = (max_width//STRIDE) * STRIDE  +  STRIDE
    final_height = (max_height//WINDOW_SIZE) * WINDOW_SIZE  +  WINDOW_SIZE
    
    pad_image = np.full((final_width , final_height , channels) , 255 , dtype=np.uint8)
    pad_image[:max_width,:max_height,:] = image
    return(pad_image)

def image_proc_lev1(img):

    
    rows = img.shape[0]
    cols = img.shape[1]
    cols_needed = []
    rows_needed = []    
    j = 0 
    k = 0 
    for j in range(cols-1):
        #if np.equal(img[:,j,0] , img[:,j+1,0]).sum() < .95 * rows:
        if np.equal(img[:,j,0] , img[:,j+1,0]).sum() != rows:
            cols_needed.append(j)

    for k in range(rows-1):
        #if np.equal(img[k,:,0] , img[k+1,:,0]).sum() < .95 * cols:
        if np.equal(img[k,:,0] , img[k+1,:,0]).sum() != cols:
            rows_needed.append(k) 
    img_new = img[:,cols_needed, :]
    img_new = img_new[rows_needed , : , : ]

#    print('image_proc_3',img_new.shape)
    h, w, d = img_new.shape
    if (h * w) <10:
        return img
    if h < w: 
        img_new = np.rot90(img_new)
#    print('image_proc_2',img_new.shape)    
    
    return  img_new

def image_proc_lev2(img , window_size , stride):
    rows_needed = [] 
    max_height = img.shape[0]
    max_width = img.shape[1]
    
    
    i = 0
    
    while window_size + window_size * i <= max_height:
        j = 0
        row_counter = 0
        row_container = []
        while window_size + stride*j <= max_width:            
            x_top_left_pixel = i * window_size
            y_top_left_pixel = j * stride
            
            patch = img[
                x_top_left_pixel : x_top_left_pixel + window_size,
                y_top_left_pixel : y_top_left_pixel + window_size,
                :
            ]
            
            ratio_white_pixels, green_concentration, blue_concentration = compute_statistics(patch)
            row_container.append(ratio_white_pixels)
            if ratio_white_pixels <= .9:
                row_counter= row_counter + 1
            j += 1
        if row_counter > 0:
            rows_needed.extend(range(window_size * i, window_size + window_size * i))
            
        i += 1    

    img_new = img[rows_needed , : , : ]

#    print('image_proc_3',img_new.shape)
#     h, w, d = img_new.shape
#     if h < w: 
#         img_new = np.rot90(img_new)
#    print('image_proc_2',img_new.shape)    
    
    return  img_new

def get_k_best_regions(coordinates, image, window_size=512):
    regions = {}
    for i, tup in enumerate(coordinates):
        x, y = tup[0], tup[1]
        regions[i] = image[x : x+window_size, y : y+window_size, :]
    
    return regions

def glue_to_one_picture(image_patches, window_size=200, k=16):
    side = int(np.sqrt(k))
    image = np.zeros((side*window_size, side*window_size, 3), dtype=np.int16)
        
    for i, patch in image_patches.items():
        x = i // side
        y = i % side
        image[
            x * window_size : (x+1) * window_size,
            y * window_size : (y+1) * window_size,
            :
        ] = patch
    
    return image


class PANDADataset(Dataset):
    def __init__(self,
                 df,
                 image_size,
                 n_tiles=n_tiles,
                 tile_mode=0,
                 rand=False,
                 transform=None,
                 tile_method = 2,
                ):

        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.n_tiles = n_tiles
        self.tile_mode = tile_mode
        self.rand = rand
        self.transform = transform
        self.tile_method = tile_method

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = row.image_id
        
        tiff_file = os.path.join(image_folder, f'{img_id}.tiff')
        im = skimage.io.MultiImage(tiff_file)[LAYER]
        
        if self.tile_method==2:
            arr = np.asarray(im)
            arr = enhance_image(arr)
            arr = unsharp_masking(arr)
            images = akensert_tiles(arr, debug=False)
        elif self.tile_method==1:
            tiles, OK = get_tiles(im, self.tile_mode)
            if self.rand:
                idxes = np.random.choice(list(range(self.n_tiles)), self.n_tiles, replace=False)
            else:
                idxes = list(range(self.n_tiles))

            n_row_tiles = int(np.sqrt(self.n_tiles))
            images = np.zeros((image_size * n_row_tiles, image_size * n_row_tiles, 3))
            for h in range(n_row_tiles):
                for w in range(n_row_tiles):
                    i = h * n_row_tiles + w

                    if len(tiles) > idxes[i]:
                        this_img = tiles[idxes[i]]['img']
                    else:
                        this_img = np.ones((self.image_size, self.image_size, 3)).astype(np.uint8) * 255
                    this_img = 255 - this_img
                    if self.transform is not None:
                        this_img = self.transform(image=this_img)['image']
                    h1 = h * image_size
                    w1 = w * image_size
                    images[h1:h1+image_size, w1:w1+image_size] = this_img
        elif self.tile_method==3:
            if white_img_flag(im)=='N':
                clip_image_lev1 = image_proc_lev1(im)
                pad_image = padding_image(clip_image_lev1)
                clip_image_lev2 = image_proc_lev2(pad_image, window_size = WINDOW_SIZE , stride = STRIDE )
                images = get_patch_image(clip_image_lev2)
            else:
                images = np.zeros([image_size*ROWS,image_size*ROWS,3],dtype=np.uint8)
                images[:] = 255            
                       
        if self.transform is not None:
            images = self.transform(image=images)['image']
        images = images.astype(np.float32)
        images /= 255
        images = images.transpose(2, 0, 1)
        white_flag = white_img_flag(images)

        return torch.tensor(images), img_id , white_flag
    
model_files = [
{'type':'split','tile_method':2,'backbone':'efficientnet-b2','dp':'yes','weight_file_k':'../input/enet-weights/effnet_new_tile_20fold_dp_K_b2_best_fold0_0719.pth','weight_file_r':'../input/enet-weights/effnet_new_tile_20fold_dp_R_b2_best_fold0_0719.pth'},
{'type':'split','tile_method':2,'backbone':'efficientnet-b2','dp':'yes','weight_file_k':'../input/enet-weights/effnet_new_tile_20fold_dp_K_b2_uplbl_best_fold0_0722.pth','weight_file_r':'../input/enet-weights/effnet_new_tile_20fold_dp_R_b2_uplbl_best_fold0_0722.pth'},
{'type':'split','tile_method':3,'backbone':'efficientnet-b2','dp':'yes','weight_file_k':'../input/enet-weights/effnet_pj_tile_20fold_dp_K_b2_best_fold0_0720.pth','weight_file_r':'../input/enet-weights/effnet_pj_tile_20fold_dp_R_b2_best_fold0_0720.pth'},
#{'type':'all','tile_method':2,'backbone':'efficientnet-b2','dp':'yes','weight_file':'../input/enet-weights/effnet_new_tile_20fold_dp_b2_best_fold0_0718.pth'},
{'type':'all','tile_method':1,'backbone':'efficientnet-b0','dp':'no','weight_file':'../input/enet-weights/effnet_b0_best_fold0.pth'}
]


def get_model(weight_file,backbone,dp='yes',out_dim=5):
    model = enetv2(backbone, out_dim=out_dim)
    if dp == 'yes':
        model = nn.DataParallel(model, device_ids=[0])
        model.load_state_dict(torch.load(weight_file, map_location=device))
    else:
#         print('no dp')
        model.load_state_dict(torch.load(weight_file, map_location=device))
    model.to(device)
    model.eval()
    return model

# model = enetv2(backbone=enet_type, out_dim=out_dim)
# model.load_state_dict(torch.load(weights_path,map_location=device))
# model = model.to(device)

def calculate_pred(eval_dict):
    model_num = len(eval_dict)
    pred = 0.
    PREDS = []
    IMG_List = []
    for i in np.arange(df.shape[0]):
        img_id = df.iloc[i].image_id
        IMG_List.append(img_id)
        temp1 = []
        
        for j in np.arange(len(eval_dict)):
            temp1.append(eval_dict[j][img_id])
            
        temp1 = torch.stack(temp1)
        temp2 = temp1.sigmoid().cpu().sum()/model_num
        pred = temp2.round().numpy()
        PREDS.append(pred.tolist())
    return IMG_List, PREDS


if os.path.exists(image_folder):
    model_num = 0
    eval_dict = {}
    for model_dict in model_files:
        backbone = model_dict['backbone']
        dp = model_dict['dp']
        LOGITS = []
        LOGITS2 = []
        LOGITS_TMP = []
        IMG = []
        IMG2 = []
    #     n_tiles = model_dict['n_tiles']
        tile_method = model_dict['tile_method']
        model_type = model_dict['type']
#         print(model_type,dp)

        if model_type == 'split':
            weight_file_k = model_dict['weight_file_k']
            weight_file_r = model_dict['weight_file_r']
            model_K = get_model(weight_file_k,backbone,dp,out_dim)
            model_R = get_model(weight_file_r,backbone,dp,out_dim)
        elif model_type == 'all':
            weight_file = model_dict['weight_file']
            model = get_model(weight_file,backbone,dp,out_dim)

        if tile_method == 1:
            dataset = PANDADataset(df=df, image_size=image_size, n_tiles=n_tiles, tile_mode=0, tile_method = 1)
            loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

            dataset2 = PANDADataset(df=df, image_size=image_size, n_tiles=n_tiles, tile_mode=2, tile_method = 1)
            loader2 = DataLoader(dataset2, batch_size=batch_size, num_workers=num_workers, shuffle=False)
            with torch.no_grad():
                for data, img_id, white_flag in tqdm(loader):
                    data = data.to(device)
                    logits = model(data)
                    LOGITS.append(logits)
                    IMG.extend(list(img_id))

                LOGITS = list(torch.cat(LOGITS))
                eval_dict[model_num] = dict(zip(IMG, LOGITS))
                model_num += 1

                for data, img_id, white_flag in tqdm(loader2):
                    data = data.to(device)
                    logits = model(data)
                    LOGITS2.append(logits)
                    IMG2.extend(list(img_id))

                LOGITS2 = list(torch.cat(LOGITS2))
                eval_dict[model_num] = dict(zip(IMG2, LOGITS2))
                model_num += 1

            LOGITS_TMP = (torch.cat(LOGITS).sigmoid().cpu() + torch.cat(LOGITS2).sigmoid().cpu()) / 2

        elif (tile_method == 2 or tile_method == 3) and model_type == 'split':
            dataset_K = PANDADataset(df=df_K, image_size=image_size, n_tiles=n_tiles, transform=None, tile_method = tile_method )
            loader_K = torch.utils.data.DataLoader(dataset_K, batch_size=batch_size, num_workers=num_workers)

            dataset_R = PANDADataset(df=df_R, image_size=image_size, n_tiles=n_tiles, transform=None, tile_method = tile_method)
            loader_R = torch.utils.data.DataLoader(dataset_R, batch_size=batch_size, num_workers=num_workers)

            with torch.no_grad():
                for (data, img_id, white_flag) in tqdm(loader_K):
                    data = data.to(device)
                    logits = model_K(data)
                    LOGITS.append(logits)
                    IMG.extend(list(img_id))
    #                 print(IMG, LOGITS)

                for (data, img_id, white_flag) in tqdm(loader_R):
                    data = data.to(device)
                    logits = model_R(data)
                    LOGITS.append(logits)
                    IMG.extend(list(img_id))

                LOGITS = list(torch.cat(LOGITS))
                eval_dict[model_num] = dict(zip(IMG, LOGITS))
                model_num += 1

        elif tile_method == 2 and model_type == 'all':
            dataset = PANDADataset(df=df, image_size=image_size, n_tiles=n_tiles, transform=None, tile_method = 2 )
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
            with torch.no_grad():
                for (data, img_id, white_flag) in tqdm(loader):
                    data = data.to(device)
                    logits = model(data)
                    LOGITS.append(logits)
                    IMG.extend(list(img_id))        

                LOGITS = list(torch.cat(LOGITS))    
                eval_dict[model_num] = dict(zip(IMG, LOGITS))
                model_num += 1

    IMG_List, PREDS = calculate_pred(eval_dict)
    submission = pd.DataFrame({'image_id':IMG_List,'isup_grade':PREDS})
    submission.isup_grade = submission.isup_grade.astype(int)
#     submission['isup_grade'] = submission.apply(lambda x: 0 if x['white_flag']=='Y' else x['isup_grade'], axis=1)
#     submission.drop(columns=['white_flag'],inplace=True)    
else:
    submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
    
submission.to_csv('submission.csv',index=False)
submission.head(10)
df.head()