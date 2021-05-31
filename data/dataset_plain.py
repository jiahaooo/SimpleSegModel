import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util

class DatasetPlain(data.Dataset):
    # -----------------------------------------
    # Get Img and Label
    # -----------------------------------------
    def __init__(self, opt):
        super(DatasetPlain, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels']
        self.patch_size = self.opt['patchsize']
        # ------------------------------------
        # get the path of paths_Input / paths_Label
        # ------------------------------------
        self.paths_Label = util.get_npys_paths(opt['dataroot_Label'])
        self.paths_Input = util.get_npys_paths(opt['dataroot_Input'])
        assert self.paths_Label, 'Error: paths_Label path is needed but it is empty.'
        assert self.paths_Input, 'Error: paths_Input path is needed but it is empty.'
        assert len(self.paths_Input) == len(self.paths_Label), 'L/H mismatch - {}, {}.'.format(len(self.paths_Input), len(self.paths_Label))

    def __getitem__(self, index):
        if self.opt['phase'] == 'train':
            # ------------------------------------
            # get image and label
            # ------------------------------------
            Label_path = self.paths_Label[index]
            img_label = util.labelread_training(Label_path, self.n_channels)
            Input_path = self.paths_Input[index]
            img_input = util.imread_training(Input_path, self.n_channels)
            # --------------------------------
            # random crop and s1mple augmentation
            # Test_Code is True only when debugging code
            # --------------------------------
            Test_Code = True
            if Test_Code:
                patch_Input = img_input
                patch_Label = img_label
            else:
                JayChou = 'JayChou'
                # H, W = img_label.shape
                # rnd_h = random.randint(0, max(0, H - self.patch_size))
                # rnd_w = random.randint(0, max(0, W - self.patch_size))
                # patch_Input = img_input[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
                # patch_Label = img_label[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
                # mode = np.random.randint(0, 8)
                # patch_Input, patch_Label = util.augment_img(patch_Input, mode=mode), util.augment_img(patch_Label, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy (uint8) to tensor
            # --------------------------------
            img_input, img_label = util.uint82tensor(patch_Input), util.label_uint82tensor(patch_Label)
        else:
            # ------------------------------------
            # get image and label
            # ------------------------------------
            Label_path = self.paths_Label[index]
            img_label = util.labelread_training(Label_path, self.n_channels)
            Input_path = self.paths_Input[index]
            img_input = util.imread_training(Input_path, self.n_channels)
            # --------------------------------
            # HWC to CHW, numpy (uint8) to tensor
            # --------------------------------
            img_input, img_label = util.uint82tensor(img_input), util.label_uint82tensor(img_label)

        return {'Input': img_input, 'Label': img_label, 'Input_path': Input_path, 'Label_path': Label_path}

    def __len__(self):
        return len(self.paths_Label)
