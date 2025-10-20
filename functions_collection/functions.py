import numpy as np
import glob 
import os
from PIL import Image
import random
import Example_UNet.Data_processing as dp


# function: set window level
def set_window(image,level,width):
    if len(image.shape) == 3:
        image = image.reshape(image.shape[0],image.shape[1])
    new = np.copy(image)
    high = level + width // 2
    low = level - width // 2
    # normalize
    unit = (1-0) / (width)
    new[new>high] = high
    new[new<low] = low
    new = (new - low) * unit 
    return new

# function: get first X numbers
# if we have 1000 numbers, how to get the X number of every interval numbers?
def get_X_numbers_in_interval(total_number, start_number, end_number , interval = 100):
    '''if no random pick, then random_pick = [False,0]; else, random_pick = [True, X]'''
    n = []
    for i in range(0, total_number, interval):
        n += [i + a for a in range(start_number,end_number)]
    n = np.asarray(n)
    return n


# function: find all files under the name * in the main folder, put them into a file list
def find_all_target_files(target_file_name,main_folder):
    F = np.array([])
    for i in target_file_name:
        f = np.array(sorted(glob.glob(os.path.join(main_folder, os.path.normpath(i)))))
        F = np.concatenate((F,f))
    return F

# function: find time frame of a file
def find_timeframe(file,num_of_dots,start_signal = '/',end_signal = '.'):
    k = list(file)

    if num_of_dots == 0: 
        num = [i for i,e in enumerate(k) if e== start_signal][-1]
        kk = k[num+1:]
    
    else:
        if num_of_dots == 1: #.png
            num1 = [i for i, e in enumerate(k) if e == end_signal][-1]
        elif num_of_dots == 2: #.nii.gz
            num1 = [i for i, e in enumerate(k) if e == end_signal][-2]
        num2 = [i for i,e in enumerate(k) if e== start_signal][-1]
        kk=k[num2+1:num1]


    total = 0
    for i in range(0,len(kk)):
        total += int(kk[i]) * (10 ** (len(kk) - 1 -i))
    return total

# function: sort files based on their time frames
def sort_timeframe(files,num_of_dots,start_signal = '/',end_signal = '.'):
    time=[]
    time_s=[]
    
    for i in files:
        a = find_timeframe(i,num_of_dots,start_signal,end_signal)
        time.append(a)
        time_s.append(a)
    time_s.sort()
    new_files=[]
    for i in range(0,len(time_s)):
        j = time.index(time_s[i])
        new_files.append(files[j])
    new_files = np.asarray(new_files)
    return new_files

# function: make folders
def make_folder(folder_list):
    for i in folder_list:
        os.makedirs(i,exist_ok = True)

# function: save grayscale image
def save_grayscale_image(a,save_path,normalize = True, WL = 50, WW = 100):
    I = np.zeros((a.shape[0],a.shape[1],3))
    # normalize
    if normalize == True:
        a = set_window(a, WL, WW)

    for i in range(0,3):
        I[:,:,i] = a
    
    Image.fromarray((I*255).astype('uint8')).save(save_path)

# function: patch definition:
def patch_definition(img_shape, patch_size, stride, count_for_overlap = False):
    # now assume patch_size is square in x and y, and same dimension as img in z 
    start = 0
    end = img_shape[0] - patch_size

    origin_x_list = np.arange(start, end+1, stride)

    patch_origin_list = [[origin_x_list[i], origin_x_list[j]] for i in range(0,origin_x_list.shape[0]) for j in range(0,origin_x_list.shape[0])]
    
    if count_for_overlap == False:
        return patch_origin_list, 0
    else:
        count = np.zeros(img_shape)
        for origin in patch_origin_list:
            count[origin[0] : (origin[0] + patch_size), origin[1] : (origin[1] + patch_size), :] += 1

        return patch_origin_list, count

# function: randomly sample patch origins
def sample_patch_origins(patch_origins, N, include_original_list = None):
    if isinstance(patch_origins, list) == False:
        patch_origins = patch_origins.tolist()
    origins = np.array(patch_origins)
    x_min,  x_max, y_min, y_max = np.min(origins[:,0]), np.max(origins[:,0]), np.min(origins[:,1]), np.max(origins[:,1])

    # Generate random coordinates
    pixels = [(random.randint(x_min, x_max), random.randint(y_min, y_max)) for _ in range(N)]

    if include_original_list is True:
        pixels = patch_origins + pixels

    return pixels


    
