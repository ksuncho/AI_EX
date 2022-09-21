import os,cv2,json

import numpy as np
import torch
import torch.nn as nn

import torch.utils.data as data


class LSMI(data.Dataset):
    def __init__(self,root,split,image_pool,image_size,
                 input_type='uvl',output_type=None,
                 mask_black=None,mask_highlight=None):
        self.root = root                        # dataset root
        self.split = split                      # train / val / test
        self.image_pool = image_pool            # 1 / 12 / 123
        self.mask_black = mask_black            # None or Masked value for black pixels 
        self.mask_highlight = mask_highlight    # None or Saturation value

        self.image_size = image_size
        # print(image_pool)
        # for f in os.listdir(os.path.join(root,split)):
        #     print(f.split('_'))
        #     print(f.split('_')[-2])
        self.image_list = sorted([f for f in os.listdir(os.path.join(root,split))
                                 if f.split('_')[-2] in image_pool
                                 and f.endswith(".tiff")
                                 and "gt" not in f])
        
        meta_path = os.path.join(self.root,'meta.json')
        with open(meta_path, 'r') as json_file:
            self.meta_data = json.load(json_file)

        self.input_type = input_type            # uvl / rgb
        self.output_type = output_type          # None / illumination / uv

        print("[Data]\t"+str(self.__len__())+" "+split+" images are loaded from "+root)

    def __getitem__(self, idx):
        """
        Returns
        metadata        : meta information
        input_tensor    : input image (uvl or rgb)
        gt_tensor       : GT (None or illumination or chromaticity)
        mask            : mask for undetermined illuminations (black pixels) or saturated pixels
        """

        # parse fname
        fname = self.image_list[idx]
        place, illum_count, img_id = os.path.splitext(fname)[0].split('_')

        # 1. prepare label
        illum1 = torch.tensor(self.meta_data[self.split][place][illum_count+'_'+img_id][0])
        illum_path = os.path.join(self.root,self.split,os.path.splitext(fname)[0]+'_illum.npy')
        gt_illumination = np.load(illum_path)

        # 2. prepare input
        # load 3-channel rgb tiff image
        input_path = os.path.join(self.root,self.split,fname)
        input_bgr = cv2.imread(input_path, cv2.IMREAD_UNCHANGED).astype('float32')
        input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
        input = torch.tensor(input_rgb).permute(2,0,1)

        if self.split == "test":
            return input, gt_illumination, fname
            # return input, illum1, fname
        else:
            return input, illum1, fname

    def __len__(self):
        return len(self.image_list)

def get_loader(config, split):
    dataset = LSMI(root=config.data_root,
                   split=split,
                   image_pool=config.image_pool,
                   image_size=config.image_size,
                   input_type=config.input_type,
                   output_type=config.output_type,
                   mask_black=config.mask_black,
                   mask_highlight=config.mask_highlight)
    
    if split == 'test':
        dataloader = data.DataLoader(dataset,batch_size=1,shuffle=False,
                                     num_workers=config.num_workers)
    else:
        dataloader = data.DataLoader(dataset,batch_size=config.batch_size,
                                     shuffle=True,num_workers=config.num_workers)

    return dataloader

if __name__ == "__main__":
    
    train_set = LSMI(root='GALAXY_synthetic',
                      split='train',image_pool=['12'],
                      image_size=256,input_type='uvl',output_type='uv')

    train_loader = data.DataLoader(train_set, batch_size=4, shuffle=False)

    for batch in train_loader:
        print(batch["illum1"])
        print(batch["illum2"])
        print(batch["illum3"])
        print(batch["fname"])
        print(batch["input"].shape)
        print(batch["gt"].shape)
        print(batch["mask"].shape)

        print(torch.cat((batch["illum1"],batch["illum2"]),1))
        input()