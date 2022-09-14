import json
import os

import cv2
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from dataset import RafdDataset, RafdTest
from model import Generator, Discriminator
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.nn.functional as F
from loss import GANLoss, PerceptualLoss, CrossEntropyLoss
import time
from tqdm import tqdm
import utils

batch_size = 32
nclasses = 75
norm_loss = False
loss_weight = {'recon': 1.0, 'vgg':10.0, 'mouth':0.1, 'gan': 1.0, 'classi': 0.1, 'seg': 0.1, 'total': 1.0}
# 定义数据集
dataset = RafdDataset('/data5/fxm/MakeItSmile/data/image_crop/')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
testset = RafdTest('/data5/fxm/MakeItSmile/data/RafdTest')
testloader = DataLoader(testset, batch_size=1, shuffle=True)

# 定义生成器，判别器
generator = Generator()
utils.init_net(generator)
discriminator = Discriminator()
utils.init_net(discriminator)
time1 = time.time()
generator = generator.cuda()
time2 = time.time()
discriminator = discriminator.cuda()
time3 = time.time()

# 定义优化器 和学习率控制
optim_generator = Adam(generator.parameters(), lr=0.00001, betas=(0.5, 0.99))
optim_discriminator = Adam(discriminator.parameters(), lr=0.00001, betas=(0.5, 0.99))
scheduler_generator = StepLR(optim_generator, step_size=20, gamma=0.5)
scheduler_discriminator = StepLR(optim_discriminator, step_size=20, gamma=0.9)

# 定义损失函数
loss_l1 = nn.L1Loss()
loss_gan = GANLoss(gan_mode='lsgan', device='cuda:0')
loss_vgg = PerceptualLoss(layers=['relu_1_1','relu_2_1','relu_3_1','relu_4_1','relu_5_1'],
                        weights=[1/16, 1/8, 1/4, 1/2, 1],
                          device='cuda:0')
loss_ce = CrossEntropyLoss()
logits_gt = torch.ones([batch_size, nclasses]) / nclasses
logits_gt = logits_gt.cuda()

# 设定存储位置
time_now = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
logpath = f'./log/{time_now}'
if not os.path.exists(logpath):
    os.mkdir(logpath)
loss_file_path = os.path.join(logpath, 'loss.txt')
with open(loss_file_path, 'a') as f:
    f.writelines(json.dumps(loss_weight) + '\n')

# 设定训练轮数
epochs = 400
iter = 0
for epoch in range(epochs):
    print(f'第{epoch}epoch')
    for data in tqdm(dataloader):
        iter = iter + 1
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = data[k].cuda()
        source = data['source_img']
        driving = data['driving_img']
        real_img = data['gt_img']
        heatmap_gt = data['driving_heatmap']
        attn_map = data['gt_attn_map']
        # 先训练生成器
        generator.train()
        utils.set_requires_grad(discriminator, False)
        utils.set_requires_grad(generator, True)
        logits, heatmap_pred, fake_img = generator(source, driving)
        pred_fake = discriminator(fake_img)
        optim_generator.zero_grad()
        loss_generator = {}
        loss_generator_norm = {}
        loss_generator['recon'] = loss_l1(fake_img, real_img)
        loss_generator['vgg'] = loss_vgg(fake_img, real_img)
        loss_generator['mouth'] = loss_l1(fake_img * attn_map, real_img * attn_map)
        loss_generator['classi'] = loss_ce(logits, logits_gt)
        loss_generator['seg'] = loss_l1(heatmap_pred, F.interpolate(heatmap_gt, heatmap_pred.shape[2:]))
        loss_generator['gan'] = loss_gan(pred_fake, True)
        for k, v in loss_generator.items():
            if norm_loss:
                loss_generator_norm[k] = v / v.item()
            else:
                loss_generator_norm[k] = v
        temp = 0
        for k, v in loss_generator_norm.items():
            temp += loss_weight[k] * loss_generator_norm[k]
        loss_generator_norm['total'] = temp
        loss_generator_norm['total'].backward()
        optim_generator.step()
        # 再训练判别器
        discriminator.train()
        utils.set_requires_grad(generator, False)
        utils.set_requires_grad(discriminator, True)
        pred_true = discriminator(real_img)
        pred_fake = discriminator(fake_img.detach())
        loss_discriminator = {}
        loss_discriminator_norm = {}
        loss_discriminator['gan'] = loss_gan(pred_true, True) + loss_gan(pred_fake, False)
        for k, v in loss_discriminator.items():
            if norm_loss:
                loss_discriminator_norm[k] = v / v.item()
            else:
                loss_discriminator_norm[k] = v
        temp = 0
        for k, v in loss_discriminator_norm.items():
            temp += loss_weight[k] * loss_discriminator_norm[k]
        loss_discriminator_norm['total'] = temp
        optim_discriminator.zero_grad()
        loss_discriminator_norm['total'].backward()
        optim_discriminator.step()
        # 隔20iter看一下结果
        if iter % 20 == 0:
            const = {'epoch':epoch, 'iter':iter}
            for k, v in loss_generator.items():
                loss_generator[k] = v.item()
            for k, v in loss_discriminator.items():
                loss_discriminator[k] = v.item()
            with open(loss_file_path, 'a') as f:
                f.writelines(json.dumps(const) + '\n')
                f.writelines(json.dumps(loss_generator) + '\n')
                f.writelines(json.dumps(loss_discriminator) + '\n')
                f.writelines('-------------' + '\n')
            cv2.imwrite(os.path.join(logpath, 'iter'+str(iter).zfill(6) + '_driving.jpg'),
                        utils.tensor2img(driving[0]))
            cv2.imwrite(os.path.join(logpath, 'iter'+str(iter).zfill(6) + '_source.jpg'),
                        utils.tensor2img(source[0]))
            cv2.imwrite(os.path.join(logpath, 'iter'+str(iter).zfill(6) + '_gt.jpg'),
                        utils.tensor2img(real_img[0]))
            cv2.imwrite(os.path.join(logpath, 'iter'+str(iter).zfill(6) + '_generated.jpg'),
                        utils.tensor2img(fake_img[0]))

            if iter % 200 == 0:
                torch.save(generator, os.path.join(logpath, 'iter' + str(iter).zfill(6) + '_gen.pth'))
                torch.save(discriminator, os.path.join(logpath, 'iter' + str(iter).zfill(6) + '_dis.pth'))
                save_dir = os.path.join(logpath, f'iter{str(iter).zfill(6)}_image')
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                generator.eval()
                data = testset[0]
                test_driving = data['driving_img'].unsqueeze(0).cuda()
                test_source = data['source_img'].unsqueeze(0).cuda()
                test_gt = data['gt_img'].unsqueeze(0).cuda()
                _, _, test_fake = generator(test_source, test_driving)
                cv2.imwrite(os.path.join(save_dir, 'driving.jpg'), utils.tensor2img(test_driving[0]))
                cv2.imwrite(os.path.join(save_dir, 'source.jpg'), utils.tensor2img(test_source[0]))
                cv2.imwrite(os.path.join(save_dir, 'gt.jpg'), utils.tensor2img(test_gt[0]))
                cv2.imwrite(os.path.join(save_dir, 'fake.jpg'), utils.tensor2img(test_fake[0]))
    scheduler_generator.step()
    scheduler_discriminator.step()
    if epoch == 200:
        loss_weight = {'recon': 10.0, 'vgg':1.0, 'mouth':10.0, 'gan': 1.0, 'classi': 0.1, 'seg': 0.1, 'total': 1.0}
        with open(loss_file_path, 'a') as f:
            f.writelines(json.dumps(loss_weight) + '\n')



