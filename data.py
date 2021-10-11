import h5py
from BCDdataset import listDataset
import torch
import numpy as np
from options import args
import os
from PIL import Image
import torchvision.transforms as transforms





def load_data(img_path):
    # gt_path = img_path.replace('.jpg', '.h5').replace('images', 'gt_density_map')
    dir_name, image_name = os.path.dirname(img_path), os.path.basename(img_path)

    gt_posititive = os.path.join(dir_name.replace('images', 'annotations'), 'positive', image_name).replace('.png', '.h5')
    gt_negative = os.path.join(dir_name.replace('images', 'annotations'), 'negative', image_name).replace('.png', '.h5')
    img = Image.open(img_path).convert('RGB')

    while True:
        try:
            gt_pfile = h5py.File(gt_posititive)
            gt_nfile = h5py.File(gt_negative)
            gt_pcoords = np.asarray(gt_pfile['coordinates'])
            gt_ncoords = np.asarray(gt_nfile['coordinates'])
            gt_pcount = len(gt_pcoords)
            gt_ncount = len(gt_ncoords)
            break  # Success!
        except OSError:
            print("load error:", img_path)
            cv2.waitKey(1000)  # Wait a bit

    img = img.copy()

    return img, gt_pcount, gt_ncount

def pre_data(train_list, args, train):
    print("Pre_load dataset ......")
    data_keys = {}
    count = 0
    for j in range(len(train_list)):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        img, gt_pcount, gt_ncount = load_data(Img_path)

        blob = {}
        blob['img'] = img

        blob['gt_pcount'] = gt_pcount
        blob['gt_ncount'] = gt_ncount
        blob['fname'] = fname
        data_keys[count] = blob
        count += 1
        '''for debug'''
        # if j> 10:
        #     break
    return data_keys

def make_dataloader():
    train_file = '/data2/Public_dataset/Public_KI67_Datasets/BCData/npydata/BCD_train.npy'
    val_file = '/data2/Public_dataset/Public_KI67_Datasets/BCData/npydata/BCD_validation.npy'
    test_file= '/data2/Public_dataset/Public_KI67_Datasets/BCData/npydata/BCD_test.npy'

    with open(train_file, 'rb') as outfile:
        train_list = np.load(outfile).tolist()
    with open(val_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()
    with open(test_file, 'rb') as outfile:
        test_list = np.load(outfile).tolist()

    train_data = pre_data(train_list, args, train=True)
    val_data = pre_data(val_list,args,train=False)
    test_data = pre_data(test_list, args, train=False)

    train_loader = torch.utils.data.DataLoader(
        listDataset(train_data,
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.Resize((224, 224), interpolation=2),


                            ]),
                            train=True,
                            batch_size=args.batch_size,
                            num_workers=args.workers,
                            args=args),
        batch_size=args.batch_size, drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        listDataset(val_data,
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.Resize((224, 224), interpolation=2),

                            ]),
                            train=False,
                            batch_size=args.batch_size,
                            num_workers=args.workers,
                            args=args),
        batch_size=args.batch_size, drop_last=False)

    test_loader = torch.utils.data.DataLoader(
        listDataset(test_data,
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.Resize((224, 224), interpolation=2),


                            ]),
                            train=False,
                            batch_size=args.batch_size,
                            num_workers=args.workers,
                            args=args),
        batch_size=args.batch_size, drop_last=False)
    return train_loader, val_loader, test_loader