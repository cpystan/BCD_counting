from tensorboardX import SummaryWriter
from utils.ColorConvolution import ColorDeconvolution
from PIL import Image
import numpy as np
from options import args
from model.swintrans import swin_base_gap
from model.ViT import ViT
from utils.optimization import make_optimizer
import torch
import torch.nn as nn
import data
import os
from tqdm import tqdm
import random

import warnings
warnings.filterwarnings('ignore')

writer = SummaryWriter()
torch.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.set_gpu
EPOCH = args.epochs


def save_model(dict,epoch,name):
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')

    torch.save(dict,f'checkpoint/{name}_{epoch}.pth')




if __name__ == "__main__":
    train_loader, val_loader, test_loader = data.make_dataloader()
    model = swin_base_gap(pretrained=True, variant = 'swin_base_patch4_window12_384_in22k').cuda()

    if args.check_path:
        model.load_state_dict(torch.load(args.check_path))

    optimizer = make_optimizer(args,model)
    loss_func = nn.L1Loss()

    if args.mode == 'train':
        model.train()
        for epoch in range(EPOCH):
            print(F'--EPOCH : {epoch} ')
            loss_list = []
            for name, img, pcount ,ncount in tqdm(train_loader,desc='trianing'):
                img = img.float().cuda()
                pcount = pcount.unsqueeze(1).float().cuda()
                ncount = ncount.unsqueeze(1).float().cuda()
                target = [pcount, ncount]

                if random.random()>0.5:
                    index = 1
                else:
                    index = 0
                output = model(img)

                #需要增加一个随机概率选择哪一个tail作为输出
                loss = loss_func(output,target[index])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_list.append(loss)

            loss_mean = sum(loss_list) / len(loss_list)
            writer.add_scalar('train/mean_loss', loss_mean, epoch)
            print(F'loss_train: {loss}')

            if (epoch+1)%args.test_interval == 0 :
                print('validating:')
                print('-----------------------')

                model.eval()
                with torch.no_grad():
                    loss1_list = []
                    loss2_list = []
                    for name, img, pcount, ncount in tqdm(test_loader, desc='validating'):
                        img = img.float().cuda()
                        pcount = pcount.unsqueeze(1).float().cuda()
                        ncount = ncount.unsqueeze(1).float().cuda()

                        output = model(img)

                        # 需要增加一个随机概率选择哪一个tail作为输出
                        loss1 = loss_func(output[0], pcount)
                        loss2 = loss_func(output[1], ncount)


                        loss1_list.append(loss1)
                        loss2_list.append(loss2)
                loss_eval1 = sum(loss1_list)/len(loss1_list)
                loss_eval2 = sum(loss2_list) / len(loss2_list)
                writer.add_scalar('eval/mean_loss_p',loss_eval1,epoch)
                writer.add_scalar('eval/mean_loss_n', loss_eval2, epoch)
                print(F'eval/mean_loss_positive: {loss_eval1}  |  eval/mean_loss_negative: {loss_eval2}')

                save_model(model.state_dict(),epoch,args.model)
            writer.add_scalar('lr',optimizer.get_lr(),epoch)
            optimizer.schedule()


    elif args.mode == 'test':
        print('testing:')
        print('-----------------------')

        model.eval()
        with torch.no_grad():
            loss1_list = []
            loss2_list = []
            for name, img, pcount, ncount in tqdm(test_loader, desc='testing'):
                img = img.float().cuda()
                pcount = pcount.unsqueeze(1).float().cuda()
                ncount = ncount.unsqueeze(1).float().cuda()

                output = model(img)

                # 需要增加一个随机概率选择哪一个tail作为输出
                loss1 = loss_func(output[0], pcount)
                loss2 = loss_func(output[1], ncount)

                loss1_list.append(loss1)
                loss2_list.append(loss2)
            loss_eval1 = sum(loss1_list) / len(loss1_list)
            loss_eval2 = sum(loss2_list) / len(loss2_list)
        loss_eval = sum(loss_list) / len(loss_list)
        writer.add_scalar('test/mean_loss', loss_eval, epoch)

        print(F'test/mean_loss: {loss_eval}')
    else:
        raise NotImplementedError(F"process mode  be train/tes, not {args.mode}.")


'''
    img = x[1,:,:,:]


    satin = ColorDeconvolution(img)
    satin.RGB_2_OD()
    image = satin.separateStain()
    a = satin.stains[:,:,2]
    #satin.showStains()
'''
