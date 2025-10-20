# dataset classes

import os
import numpy as np
import nibabel as nb
import random
from scipy import ndimage
from skimage.measure import block_reduce

import torch
from torch.utils.data import Dataset
import Example_UNet.Data_processing as Data_processing
import Example_UNet.functions_collection as ff


# random function
def random_rotate(i, z_rotate_degree = None, z_rotate_range = [-10,10], fill_val = None, order = 1):
    # only do rotate according to z (in-plane rotation)
    if z_rotate_degree is None:
        z_rotate_degree = random.uniform(z_rotate_range[0], z_rotate_range[1])

    if fill_val is None:
        fill_val = np.min(i)
    
    if z_rotate_degree == 0:
        return i, z_rotate_degree
    else:
        if len(i.shape) == 2:
            return Data_processing.rotate_image(np.copy(i), z_rotate_degree, order = order, fill_val = fill_val, ), z_rotate_degree
        else:
            return Data_processing.rotate_image(np.copy(i), [0,0,z_rotate_degree], order = order, fill_val = fill_val, ), z_rotate_degree

def random_translate(i, x_translate = None,  y_translate = None, translate_range = [-10,10]):
    # only do translate according to x and y
    if x_translate is None or y_translate is None:
        x_translate = int(random.uniform(translate_range[0], translate_range[1]))
        y_translate = int(random.uniform(translate_range[0], translate_range[1]))
    
    if len(i.shape) == 2:
        return Data_processing.translate_image(np.copy(i), [x_translate,y_translate]), x_translate,y_translate
    else:
        return Data_processing.translate_image(np.copy(i), [x_translate,y_translate,0]), x_translate,y_translate


class Dataset_2D(Dataset):
    def __init__(
        self,
        input_list,
        reference_list,

        image_size,

        num_slices_per_image,
        random_pick_slice,
        slice_range, # None or [a,b]

        background_cutoff, 
        maximum_cutoff,
        normalize_factor,

        num_patches_per_slice = None,
        patch_size = None,

        shuffle = False,
        augment = False,
        augment_frequency = 0,
    ):
        super().__init__()
        self.input_list = input_list
        self.reference_list = reference_list
        self.image_size = image_size

        self.num_slices_per_image = num_slices_per_image
        self.random_pick_slice = random_pick_slice
        self.slice_range = slice_range
        self.num_patches_per_slice = num_patches_per_slice
        self.patch_size = patch_size

        self.background_cutoff = background_cutoff
        self.maximum_cutoff = maximum_cutoff
        self.normalize_factor = normalize_factor

        self.shuffle = shuffle
        self.augment = augment
        self.augment_frequency = augment_frequency
        self.num_files = len(input_list)

        self.index_array = self.generate_index_array()
        self.current_input_file = None
        self.current_input_data = None
        self.current_reference_file = None
        self.current_reference_data = None

    def generate_index_array(self):
        np.random.seed()
        index_array = []; index_array_patches = []
        
        if self.shuffle == True:
            f_list = np.random.permutation(self.num_files)
        else:
            f_list = np.arange(self.num_files)

        for f in f_list:
            s_list = np.arange(self.num_slices_per_image)
            for s in s_list:
                index_array.append([f, s])
                if self.num_patches_per_slice != None:
                    patch_list = np.arange(self.num_patches_per_slice)
                    for p in patch_list:
                        index_array_patches.append([f, s,p])
        if self.num_patches_per_slice != None:
            return index_array_patches
        else:
            return index_array

    def __len__(self):
        if self.num_patches_per_slice != None:
            return self.num_files * self.num_slices_per_image * self.num_patches_per_slice
        else:
            return self.num_files * self.num_slices_per_image
    

    def load_file(self, filename):
        ii = nb.load(filename).get_fdata()

        # cutoff and normalization
        ii = Data_processing.cutoff_intensity(ii,cutoff_low = self.background_cutoff, cutoff_high = self.maximum_cutoff)
        ii = Data_processing.normalize_image(ii, normalize_factor = self.normalize_factor, image_max = self.maximum_cutoff, image_min = self.background_cutoff ,invert = False)
        ii = Data_processing.crop_or_pad(ii, [self.image_size[0], self.image_size[1], ii.shape[2]], value= np.min(ii))

        return ii
        
    def __getitem__(self, index):
        if self.num_patches_per_slice != None:
            f,s,p = self.index_array[index]
        else:
            f,s = self.index_array[index]
        # print('index is: ', index, ' now we pick file ', f)
        input_filename = self.input_list[f]
        reference_filename = self.reference_list[f]
        # print('input filename is: ', input_filename, ' and reference filename is: ', reference_filename)

        if input_filename != self.current_input_file:
            # load input
            img = self.load_file(input_filename)
            # print('load image file: ', input_filename, ' with shape: ', img.shape)
            self.current_input_file = input_filename
            self.current_input_data = np.copy(img)

            # load reference
            ref = self.load_file(reference_filename)
            # print('load reference file: ', reference_filename, ' with shape: ', ref.shape)
            self.current_reference_file = reference_filename
            self.current_reference_data = np.copy(ref)

            # define a list of random slice numbers
            if self.slice_range == None:
                total_slice_range =  [0 ,self.current_input_data.shape[2]]
            else:
                total_slice_range = self.slice_range
            # print('in this condition case, total slice range is: ', total_slice_range)
            if self.random_pick_slice == False:
                self.slice_index_list = np.arange(total_slice_range[0], total_slice_range[1])
                self.slice_index_list = self.slice_index_list[:self.num_slices_per_image]
            else:
                self.slice_index_list = np.random.permutation(np.arange(total_slice_range[0], total_slice_range[1]))[:self.num_slices_per_image]
            # print('in this condition case, slice index list is: ', self.slice_index_list, ' with length: ', len(self.slice_index_list))

        # pick the slice
        # print('pick the slice: ', self.slice_index_list[s])
        s = self.slice_index_list[s]
        input_slice = self.current_input_data[:,:,s]
        reference_slice = self.current_reference_data[:,:,s]
        if self.num_patches_per_slice != None:
            x_shape, y_shape = input_slice.shape[0], input_slice.shape[1]
            random_origin_x, random_origin_y = random.randint(0, x_shape - self.patch_size[0]), random.randint(0, y_shape - self.patch_size[1])
            print('random origin x is: ', random_origin_x, ' and random origin y is: ', random_origin_y)
            input_slice = input_slice[random_origin_x: random_origin_x + self.patch_size[0], random_origin_y: random_origin_y + self.patch_size[1]]
            reference_slice = reference_slice[random_origin_x: random_origin_x + self.patch_size[0], random_origin_y: random_origin_y + self.patch_size[1]]

        # augmentation
        if self.augment == True:
            if random.uniform(0,1) < self.augment_frequency:
                input_slice, z_rotate_degree = random_rotate(input_slice , order = 1)
                reference_slice, _ = random_rotate(reference_slice , z_rotate_degree = z_rotate_degree, order = 1)
                input_slice, translate, y_translate = random_translate(input_slice)
                reference_slice, _, _ = random_translate(reference_slice, x_translate = translate, y_translate = y_translate)
                print('z rotate degree is: ', z_rotate_degree, ' translate is: ', translate)

        input_data = torch.from_numpy(input_slice).unsqueeze(0).float()
        output_data = torch.from_numpy(reference_slice).unsqueeze(0).float()
        # print('input data shape is: ', input_data.shape, ' and output data shape is: ', output_data.shape)
        return input_data, output_data
        
    
    def on_epoch_end(self):
        print('now run on_epoch_end function')
        self.index_array = self.generate_index_array()
    
