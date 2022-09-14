import random
import time

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import os
from torchvision import transforms
import cv2
from face_parsing.model import BiSeNet

class RafdDataset(Dataset):
    def __init__(self, root, device='cuda:0'):
        self.root = root
        self.device = device
        self.mode = 'shelf' # 'fly'
        self.filenames = os.listdir(root)
        self.temp = []
        for filename in self.filenames:
            if 'parsing' not in filename:
                self.temp.append(filename)
        self.filenames = self.temp
        # 测试集是所有大于62的id
        self.id_test = 62
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.face_parsing_net = BiSeNet(n_classes=19)
        self.face_parsing_net.load_state_dict(torch.load('./face_parsing/79999_iter.pth', map_location='cpu'))
        self.face_parsing_net.to(device)
        self.face_parsing_net.eval()

    def getAttri(self, filename):
        filename = os.path.splitext(filename)[0]
        attri_list = filename.split('_')
        attri_dict = {}
        attri_dict['id'] = attri_list[1]
        attri_dict['race'] = attri_list[2]
        attri_dict['gender'] = attri_list[3]
        attri_dict['emotion'] = attri_list[4]
        attri_dict['gaze'] = attri_list[5]
        return attri_dict

    def getImage(self, filepath):
        img = cv2.imread(filepath)
        img = cv2.resize(img, [256, 256])
        img = self.transform(img)
        return img

    # 用人脸解析网络进行解析 嘴部赋予10 其余位置赋予1
    def getHeatmap(self, image):
        if self.mode == 'shelf':
            pass
        else:
            image = image.unsqueeze(0)
            input = F.interpolate(image, [512,512], mode='bilinear', align_corners=True)
            feat_out = self.face_parsing_net(input)[0]
            feat_out = F.interpolate(feat_out, [256,256], mode='bilinear', align_corners=True)
            # 获得人脸每一部分的标签
            parsing = torch.argmax(feat_out, dim=1)
            heatmap = torch.ones_like(parsing)
            tens = torch.ones_like(parsing) * 10
            heatmap = torch.where(parsing==11, tens, heatmap)
            heatmap = torch.where(parsing==12, tens, heatmap)
            heatmap = torch.where(parsing==13, tens, heatmap)
        return heatmap

    def getAttnmap(self, image):
        if self.mode == 'shelf':
            pass
        else:
            image = image.unsqueeze(0)
            input = F.interpolate(image, (512,512), mode='bilinear', align_corners=True)
            feat_out = self.face_parsing_net(input)[0]
            feat_out = F.interpolate(feat_out, (256,256), mode='bilinear', align_corners=True)
            # 获得人脸每一部分的标签
            parsing = torch.argmax(feat_out, dim=1)
            attn_map = torch.zeros_like(parsing)
            ones = torch.ones_like(attn_map)
            attn_map = torch.where(parsing==11, ones, attn_map)
            attn_map = torch.where(parsing==12, ones, attn_map)
            attn_map = torch.where(parsing==13, ones, attn_map)
        return attn_map

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        time1 = time.time()
        source_filename = random.choice(self.filenames)
        source_id = self.getAttri(source_filename)['id']
        source_emotion = self.getAttri(source_filename)['emotion']
        gt_filename = source_filename.replace(source_emotion, 'happy')
        # 保证source_id不要取到测试集里面的，这里的测试集是id在62以上的
        while int(source_id) >= int(self.id_test) or not os.path.exists(os.path.join(self.root, gt_filename)):
            source_filename = random.choice(self.filenames)
            source_id = self.getAttri(source_filename)['id']
            source_emotion = self.getAttri(source_filename)['emotion']
            gt_filename = source_filename.replace(source_emotion, 'happy')
        driving_filename = random.choice(self.filenames)
        driving_id = self.getAttri(driving_filename)['id']
        driving_emotion = self.getAttri(driving_filename)['emotion']
        driving_filename = driving_filename.replace(driving_emotion, 'happy')
        while int(driving_id) >= int(self.id_test) or not os.path.exists(os.path.join(self.root, driving_filename)):
            driving_filename = random.choice(self.filenames)
            driving_id = self.getAttri(driving_filename)['id']
            driving_emotion = self.getAttri(driving_filename)['emotion']
            driving_filename = driving_filename.replace(driving_emotion, 'happy')
        time2 = time.time()
        source_img = self.getImage(os.path.join(self.root, source_filename)).to(self.device)
        driving_img = self.getImage(os.path.join(self.root, driving_filename)).to(self.device)
        gt_img = self.getImage(os.path.join(self.root, gt_filename)).to(self.device)
        time3 = time.time()
        if self.mode == 'shelf':
            heatmap = cv2.imread(os.path.join(self.root, ''.join(driving_filename.split('.')[:-1])+'_parsing.jpg'))
            heatmap = torch.tensor(heatmap).permute(2,0,1)[0]
            attn_map = cv2.imread(os.path.join(self.root, ''.join(gt_filename.split('.')[:-1])+'_parsing.jpg'))
            attn_map = torch.tensor(attn_map).permute(2,0,1)[0]
            tens = torch.ones_like(heatmap) * 10
            ones = torch.ones_like(heatmap)
            zeros = torch.zeros_like(heatmap)
            driving_heatmap = torch.where(heatmap>0, tens, ones).unsqueeze(0)
            gt_attn_map = torch.where(attn_map>0, ones, zeros).unsqueeze(0)
        else:
            driving_heatmap = self.getHeatmap(driving_img)
            gt_attn_map = self.getAttnmap(gt_img)
        time4 = time.time()
        output = {'driving_id': driving_id, 'source_img': source_img,
                  'driving_img': driving_img, 'gt_img': gt_img,
                  'driving_heatmap': driving_heatmap.float(),'gt_attn_map': gt_attn_map.float(),
                  'source_filename':source_filename, 'driving_filename':driving_filename,
                  'gt_filename': gt_filename}
        return output

class RafdTest(RafdDataset):
    def __init__(self, root):
        super().__init__(root)
        self.id_test = 75


def ToImage(img_tensor):
    if len(img_tensor.shape) == 4:
        B, C, H, W = img_tensor.shape
    elif len(img_tensor.shape) == 3:
        C, H, W = img_tensor.shape
    else:
        C = 1
    if C == 3:
        img_tensor = img_tensor.permute(1,2,0)
    return img_tensor.cpu().squeeze().numpy().astype('uint8')


# if __name__ == '__main__':
    # # 测试看数据集好用不
    # dataset = RafdDataset('/data5/fxm/MakeItSmile/data/image_crop/')
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    # for data in dataloader:
    #     driving_heatmap = data['driving_heatmap'][0] * 20
    #     gt_attn_map = data['gt_attn_map'][0] * 240
    #     driving_img = (data['driving_img'][0] + 1 ) * 127
    #     driving_heatmap = ToImage(driving_heatmap)
    #     gt_attn_map = ToImage(gt_attn_map)
    #     driving_img = ToImage(driving_img)
    #     gt_img= ToImage((data['gt_img'][0] + 1) * 127)
    #     cv2.imwrite('heatmap.jpg', driving_heatmap)
    #     cv2.imwrite('attn_map.jpg', gt_attn_map)
    #     cv2.imwrite('driving.jpg', driving_img)
    #     cv2.imwrite('gt.jpg',gt_img)
    #     pass






