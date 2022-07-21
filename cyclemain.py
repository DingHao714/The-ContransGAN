import torch
import argparse
import cv2
import numpy as np
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
import pandas as pd
from dataset import LiverDataset
from torch.optim import lr_scheduler
from ssim2 import ssim,msssim,SSIM,MSSSIM
from swintransformer3gray import unet,Discriminator
import itertools
from transcnn2 import SwinTransformer
from Unet import unetno
import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
from visdom import Visdom


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# x_transformsa = transforms.ToTensor()
# y_transformsa = transforms.ToTensor()
# liver_dataseta = LiverDataset("train", transform=x_transformsa, target_transform=y_transformsa)
# dataloada = torch.utils.data.DataLoader(liver_dataseta)
# num_imgs = 0
# for x, y in dataloada:
#     num_imgs += 1
#     x1 = x.mean()
#     x2 = x.std()
#     xmeans = np.asarray(x1) / num_imgs
#     xstdevs = np.asarray(x2) / num_imgs


x_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    # transforms.RandomRotation(45),  # 随机旋转，-45到45度之间随机选择一个
    # transforms.CenterCrop(size=(500, 450)),  # 从中心始裁剪,指从中间裁剪出 200*150 的图像
    # transforms.RandomHorizontalFlip(p=0.5),  # 以50%的概率随机水平翻转
    # transforms.RandomVerticalFlip(p=0.5),  # 以50%的概率随机垂直翻转
    # transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1), # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
    transforms.Normalize([0.5], [0.5])
])
y_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


parse = argparse.ArgumentParser()

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def train():
   
    epochs =60  
    netG_A = SwinTransformer().to(device)
    netG_B = SwinTransformer().to(device)
    # netG_A = unetno(1,1).to(device)
    # netG_B = unetno(1,1).to(device)
    netD_A = Discriminator(1).to(device)
    netD_B = Discriminator(1).to(device)
    batch_size =1
    #loss function
    criterion_GAN = nn.MSELoss()  
    # L1 loss
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    lr = 0.0001
    optimizer_G = optim.Adam(itertools.chain(netG_A.parameters(), netG_B.parameters()), lr=lr)
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.999))
    
    scheduler_netG = lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.95)
    scheduler_netD_A = lr_scheduler.StepLR(optimizer_D_A, step_size=10, gamma=0.95)
    scheduler_netD_B = lr_scheduler.StepLR(optimizer_D_B, step_size=10, gamma=0.95)
    liver_dataset = LiverDataset("trainqx", transform=x_transforms, target_transform=y_transforms)
    dataload = torch.utils.data.DataLoader(
        liver_dataset,  
        batch_size=batch_size,  
        shuffle=True,  
        num_workers=0,  
        pin_memory=True
    )
    netG_A.train()
    netG_B.train()
    netD_A.train()
    netD_B.train()

    netG_A.apply(weights_init_normal)
    netG_B.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()
    Loss_list = []
    Lossscore_list = []
    for epoch in range(epochs):
        # print('Epoch {}/{}'.format(epoch, epochs - 1))
        # print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        ssim_score=0
        a=0
        step = 0
        for x, y in dataload:

            step += 1
            real_A = x.to(device)
            real_B = y.to(device)
            
            optimizer_G.zero_grad()
            # # Identity loss

            # GAN loss
            fake_B = netG_A(real_A)
            pred_fake = netD_B(fake_B)
            target_real = torch.ones(pred_fake.size()).to(device)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netG_B(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss（cycle consistency loss ）
            recovered_A = netG_B(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

            recovered_B = netG_A(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

            # Total loss
            # loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G =  loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()

            optimizer_G.step()

            # ===================================Discriminator A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            target_fake = torch.zeros(pred_fake.size()).to(device)
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()

            optimizer_D_A.step()
            ###################################

            # ===================================Discriminator B ######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()

            optimizer_D_B.step()

            epoch_loss += loss_G.item()

            Loss_list.append(epoch_loss)

            # print("%d/%d,train_loss:%0.5f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()),ssim_value.item())
            print(loss_G.item())

            # print("%d个epoch learning：%f" % (epoch, optimizer.param_groups[0]['lr']))
        scheduler_netG.step()
        scheduler_netD_A.step()
        # print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        scheduler_netD_B.step()
        # print("epoch %d loss:%0.5f" % (epoch, epoch_loss / step),"epoch %d ssim_value :%0.5f" % (epoch, a/500))


    torch.save(netG_A.state_dict(), 'weights_%d_UNet1_lr1e-3.pth' % (epoch))
# 显示模型的输出结果
import time
def test():
    model = SwinTransformer()
    # model = unetno(1, 1)
    model.load_state_dict(torch.load(args.ckpt, map_location='cuda'))
    liver_dataset = LiverDataset("10datu", transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()  
    import matplotlib.pyplot as plt
    plt.ion()
    count = 0
    ssimlist=[]
    with torch.no_grad():
        for x,y in dataloaders:
            time0 = time.time()
            out = model(x)
            # torch.cuda.synchronize()
            time1 = time.time()
            print(time1 - time0)
            img_y = torch.squeeze(out).numpy()
            ssimscore=ssim(out,y)
            # ssimscore=ssimscore.data.item()
            print(ssimscore.item())
            ssimlist.append(ssimscore.item())
            # img_y = cv2.normalize(img_y, img_y, 0, 1, cv2.NORM_MINMAX)
            #plt.imshow(img_y, "gray")
            # img_y = np.array(img_y) * 255  ##img_mask为二值图
            # img_y = cv2.merge(img_y)
            # img_y = (img_y * 255).astype(np.uint8)  ##img_mask为RGB
            # out = img_y[0]
            # plt.imshow(out, "gray")
            count += 1
            img_path = '10datuout/%04dout.png' % count
            # plt.imsave(img_path, out / np.max(out) * 255)
            cv2.imwrite(img_path, img_y * 255)
            # cv2.imwrite(img_path, img_y[0] / np.max(img_y[0]) * 255)
            #plt.pause(0.01)
            
            
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--action", type=str, help="train or test", default="test")
    parse.add_argument("--batch_size", type=int, default=4)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file", default="weights_70_UNet1_lr1e-3.pth")
    args = parse.parse_args()

    if args.action == "train":
        train()
    elif args.action == "test":
        test()
