from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

from torch.utils.data import Dataset
from PIL import Image


class Self_datasets(Dataset):
    def __init__(self, path_Data, config, train=True, test=False):
        super(Self_datasets, self)

        if train:
            images_list = sorted(os.listdir(path_Data+'train/images/'))
            masks_list = sorted(os.listdir(path_Data+'train/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'train/images/' + images_list[i]
                mask_path = path_Data+'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.train_transformer
        else:
            if test:
                images_list = sorted(os.listdir(path_Data+'test/images/'))
                masks_list = sorted(os.listdir(path_Data+'test/masks/'))
                self.data = []
                for i in range(len(images_list)):
                    img_path = path_Data+'test/images/' + images_list[i]
                    mask_path = path_Data+'test/masks/' + masks_list[i]
                    self.data.append([img_path, mask_path])
                self.transformer = config.test_transformer
            else:
                images_list = sorted(os.listdir(path_Data+'val/images/'))
                masks_list = sorted(os.listdir(path_Data+'val/masks/'))
                self.data = []
                for i in range(len(images_list)):
                    img_path = path_Data+'val/images/' + images_list[i]
                    mask_path = path_Data+'val/masks/' + masks_list[i]
                    self.data.append([img_path, mask_path])
                self.transformer = config.val_transformer
        
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))

        return img, msk

    def __len__(self):
        return len(self.data)