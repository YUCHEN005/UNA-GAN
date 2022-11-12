import os.path
from data.base_dataset import BaseDataset, get_transform
from data.dataset_folder import make_dataset
from PIL import Image
import random
import util.util as util
import util.functions as functions
import numpy as np
import os
import torch
from joblib import Parallel, delayed
import multiprocessing
import subprocess
from itertools import chain
from collections import OrderedDict
import math

def split_and_save(spec, pow=1.0, state = "Train", channels = 1):
    """
        Info: Takes a spectrogram, splits it into equal parts; uses median padding to achieve this.
        Created: 13/04/2021
        By: Shashank S Shirol
        Parameters:
            spec - Magnitude Spectrogram
            pow - value to raise the spectrogram by
            phase - Decides how the components are returned
    """
    fix_w = 128  # because we have 129 n_fft bins; this will result in 129x128 spec components
    orig_shape = spec.shape

    #### adding the padding to get equal splits
    w = orig_shape[1]
    mod_fix_w = w % fix_w
    extra_cols = 0
    if(mod_fix_w != 0):
        extra_cols = fix_w - mod_fix_w
        
    #making padding by repeating same audio (takes care of edge case where actual data < padding columns to be added)
    num_wraps = math.ceil(extra_cols/w)
    temp_roll = np.tile(spec, num_wraps)
    padd=temp_roll[:,:extra_cols]
    spec = np.concatenate((spec, padd), axis=1)
    ####

    spec_components = []

    spec = functions.power_to_db(spec**pow)
    X, X_min, X_max = functions.scale_minmax(spec, 0, 255)
    X = np.flip(X, axis=0)
    np_img = X.astype(np.uint8)

    curr = [0]
    while(curr[-1] < w):
        temp_spec = np_img[:, curr[-1]:curr[-1] + fix_w]
        rgb_im = functions.to_rgb(temp_spec, chann = channels)
        img = Image.fromarray(rgb_im)
        spec_components.append(img)
        curr.append(curr[-1] + fix_w)

    if(state == "Train"):
        return spec_components if extra_cols == 0 else spec_components[:-1]  # No need to return the component with padding.
    else:
        return spec_components  # If in "Test" state, we need all the components

# Parallize processing the spectrograms


def processInput(filepath, power, state, channels):
    mag_spec, phase, sr = functions.extract(filepath, sr=8000, energy=1.0, state = state)
    components = split_and_save(mag_spec, pow=power, state = state, channels = channels)

    return components


# out = processInput('data/test.wav', 1.0, "Train", 1)
# print(out)
# print(len(out),=======================)

def countComps(sample):
    return len(sample)


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.

    Modified: 15/04/2021 19:04 Hrs
    By: Shashank S Shirol
    Changes:This dataloader takes audio files hosted in two directories as above (instead of images).
            The code extracts spectrograms and splits them into square components and treats them as independent samples.
            The process is parallelized using threads for faster processing of the components.
            CONTRARY TO THE FILE NAME AND CLASS NAME, THIS CODE NOW WORKS FOR PAIRED SAMPLES AND UNPAIRED SAMPLES.

    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        if(opt.state == "Train"):
            self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        if(opt.state == "Train"):
            self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'

        if("passcodec" in opt.preprocess):
            print("------Passing samples through g726 Codec using FFmpeg------")
            for path in self.A_paths:
                subprocess.call(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', path, '-ar', '8k', '-y', path[:-4] + '_8k.wav'])
                subprocess.call(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', path[:-4] + '_8k.wav', '-acodec', 'g726', '-b:a', '16k', path[:-4] + '_fmt.wav'])
                subprocess.call(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', path[:-4] + '_fmt.wav', '-ar', '8k', '-y', path])
                if(os.name == 'nt'):  # Windows
                    os.system('del ' + path[:-4] + '_fmt.wav')
                    os.system('del ' + path[:-4] + '_8k.wav')
                else:  # Linux/MacOS/BSD
                    os.system('rm ' + path[:-4] + '_fmt.wav')
                    os.system('rm ' + path[:-4] + '_8k.wav')

        self.spec_power = opt.spec_power
        self.energy = opt.energy
        self.state = opt.state
        self.parallel_data = True if opt.parallel_data == 1 else False
        self.gray = True if opt.single_channel == 1 else False
        self.channels = 1 if self.gray else 3
        self.num_cores = multiprocessing.cpu_count()

        #Compute the spectrogram components parallelly to make it more efficient; uses Joblib, maintains order of input data passed.
        self.clean_specs = Parallel(n_jobs=self.num_cores, prefer="threads")(delayed(processInput)(i, self.spec_power, self.state, self.channels) for i in self.A_paths)

        #calculate no. of components in each sample
        self.no_comps_clean = Parallel(n_jobs=self.num_cores, prefer="threads")(delayed(countComps)(i) for i in self.clean_specs)
        self.clean_spec_paths = []
        self.clean_comp_dict = OrderedDict()

        for nameA, countA in zip(self.A_paths, self.no_comps_clean):  # Having an OrderedDict to access no. of components, so we can wait before generation to collect all components
            self.clean_spec_paths += [nameA] * countA
            self.clean_comp_dict[nameA] = countA

        ##To separate the components; will treat every component as an individual sample
        self.clean_specs = list(chain.from_iterable(self.clean_specs))
        self.clean_specs_len = len(self.clean_specs)
        assert self.clean_specs_len == len(self.clean_spec_paths)
        

        ##Checking what samples are loaded:
        if(self.parallel_data or self.opt.serial_batches):
            print("-------Taking Parallel Samples-------")
        else:
            print("-------Taking Non - Parallel Samples-------")

        #clearing memory
        del self.no_comps_clean

        if(self.state == "Train"): ##Preparing domainB dataset is only required if we are in the Training state; for generation, we don't require domainB
            self.noisy_specs = Parallel(n_jobs=self.num_cores, prefer="threads")(delayed(processInput)(i, self.spec_power, self.state, self.channels) for i in self.B_paths)
            self.no_comps_noisy = Parallel(n_jobs=self.num_cores, prefer="threads")(delayed(countComps)(i) for i in self.noisy_specs)
            self.noisy_spec_paths = []
            self.noisy_comp_dict = OrderedDict()
            for nameB, countB in zip(self.B_paths, self.no_comps_noisy):
                self.noisy_spec_paths += [nameB] * countB
                self.noisy_comp_dict[nameB] = countB
            self.noisy_specs = list(chain.from_iterable(self.noisy_specs))
            self.noisy_specs_len = len(self.noisy_specs)
            assert self.noisy_specs_len == len(self.noisy_spec_paths)
            del self.no_comps_noisy

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths if in 'Train' mode else only A, A_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- file paths
            B_paths (str)    -- file paths
        """

        transform = get_transform(self.opt, grayscale= self.gray)

        index_A = index % self.clean_specs_len
        A_path = self.clean_spec_paths[index_A]  # make sure index is within then range
        A_img = self.clean_specs[index_A]
        A = transform(A_img)

        if(self.state == "Train"):
            if self.opt.serial_batches or self.parallel_data:   # make sure index is within then range
                index_B = index % self.noisy_specs_len
            else:   # randomize the index for domain B to avoid fixed pairs.
                index_B = random.randint(0, self.noisy_specs_len - 1)
            B_path = self.noisy_spec_paths[index_B]
            B_img = self.noisy_specs[index_B]
            B = transform(B_img)
        


        if(self.state == "Train"):
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
        else:
            B = torch.rand(1) ## A random initialization (required to prepare the model for generation, refer models.py for more info); doesn't effect the generation process.
            return {'A': A, 'A_paths': A_path, 'A_comps': self.clean_comp_dict[A_path], "B": B}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take length of the source domain ("From" Set)
        """
        return self.clean_specs_len


