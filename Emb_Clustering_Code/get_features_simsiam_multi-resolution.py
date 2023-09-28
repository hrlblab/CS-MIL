import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model, get_backbone
from tools import AverageMeter
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
import glob
import matplotlib.pyplot as plt
import torchvision.transforms as T
import pandas as pd
import numpy as np
import random

def main(args):
    tensor_transform = T.Compose([
        T.Resize([256,]),
        # T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    resolution_list = [256,512,1024]
    for rrri in range(len(resolution_list)):

        resolution = resolution_list[rrri]

        if resolution == 1024:
            args.eval_from = 'checkpoint/simsiam-GCA-b6128s256_ori1024patch_0414131202.pth'
        elif resolution == 512:
            args.eval_from = 'checkpoint/simsiam-GCA-b6128s256_ori512patch_0418160825.pth'
        else:
            args.eval_from = 'checkpoint/simsiam-GCA-b6128s256_ori256patch_0422134301.pth'

        severe = 2

        if severe == 0:
            label_file = pd.read_csv('/Data2/GCA/simsiam/data_list.csv')
            features_output = '/Data2/GCA/simsiam/feature_data_multi-resolution/'
            data_dir = '/media/dengr/Seagate Backup Plus Drive/GCA_multi-resolution/GCA_Original_Series_patch_multi-resolution_0421'

        elif severe == 1:
            label_file = pd.read_csv('/Data2/GCA/data_list_severe.csv')
            features_output = '/Data2/GCA/simsiam/feature_data_multi-resolution_severe/'
            data_dir = '/media/dengr/Seagate Backup Plus Drive/GCA_multi-resolution/GCA_Original_Series_patch_multi-resolution_0421'
        else:
            label_file = pd.read_csv('/Data2/GCA/data_list_severe_extend.csv')
            features_output = '/Data2/GCA/simsiam/feature_data_multi-resolution_severe_extend/'
            data_dir = '/media/dengr/Seagate Backup Plus Drive/GCA_multi-resolution/GCA_Original_Series_patch_multi-resolution_extend_0621'

        images = []

        model = get_backbone(args.model.backbone)

        assert args.eval_from is not None
        save_dict = torch.load(args.eval_from, map_location='cpu')
        msg = model.load_state_dict({k[9:]:v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')}, strict=True)

        # print(msg)
        model = model.to(args.device)
        model = torch.nn.DataParallel(model)

        model.eval()

        for row in range(len(label_file)):
            now_case = label_file.iloc[row]['filename'].replace('.svs','')
            now_images = glob.glob(os.path.join(data_dir, now_case, '*size%d.png' % (resolution)))
            images.extend((now_images))

        random.shuffle(images)
        patch_size = resolution
        batch = 32 #int(args.eval.batch_size)
        bag_num = int(len(images) / batch) + 1

        if not os.path.exists(features_output):
            os.makedirs(features_output)

        for ri in range(bag_num):
            if ri != bag_num - 1:
                now_images = images[ri * batch : (ri + 1) * batch]
            else:
                now_images = images[ri * batch:]

            tensor = np.zeros((len(now_images), patch_size , patch_size, 3))
            for ni in range(len(now_images)):
                image_folder = now_images[ni]
                tensor[ni] = plt.imread(image_folder)[:,:,:3]

            # tensor = tensor.transpose([0,3,1,2])
            tensor = torch.from_numpy(tensor).permute([0,3,1,2])
            inputs = tensor_transform(tensor)
            features = model(inputs.to(args.device).float())

            for fi in range(len(features)):
                now_name = os.path.basename(now_images[fi])
                wsi_name = now_name.split('_')[0]
                now_feature = features[fi].detach().cpu().numpy()
                save_root = os.path.join(features_output, wsi_name)

                if not os.path.exists(save_root):
                    os.makedirs(save_root)

                save_dir = os.path.join(save_root, now_name.replace('.png', '.npy'))
                np.save(save_dir, now_feature)


if __name__ == "__main__":
    main(args=get_args())
















