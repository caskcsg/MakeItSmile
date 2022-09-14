import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
from face_parsing.model import BiSeNet
import cv2
import torch.nn.functional as F

device = 'cuda:0'
model = BiSeNet(n_classes=19)
model.load_state_dict(torch.load('./face_parsing/79999_iter.pth'))
model.to(device)
model.eval()

root = '/data5/fxm/MakeItSmile/data/image_crop'
filenames = os.listdir(root)
filepaths = [os.path.join(root, filename) for filename in filenames]
for filepath in filepaths:
    image = cv2.imread(filepath)
    image = torch.tensor(image).float().permute(2,0,1).unsqueeze(0).to(device)
    image = ((image / 255.0) - 0.5) * 2
    image = F.interpolate(image, [512, 512], mode='bilinear', align_corners=True)
    feat_out = model(image)[0]
    feat_out = F.interpolate(feat_out, [256, 256], mode='bilinear', align_corners=True)
    parsing = torch.argmax(feat_out, dim=1)
    heatmap = torch.zeros_like(parsing)
    ones = torch.ones_like(parsing)
    heatmap = torch.where(parsing==11, ones, heatmap)
    heatmap = torch.where(parsing==12, ones, heatmap)
    heatmap = torch.where(parsing==13, ones, heatmap)
    heatmap = heatmap * 255.0
    heatmap = heatmap.detach().cpu().squeeze().numpy().astype('uint8')
    save_path = filepath.split('.')[0] + '_parsing.jpg'
    cv2.imwrite(save_path, heatmap)
    # img = cv2.imread(save_path)
    # img = torch.tensor(img).float().permute(2,0,1)
    pass
