import os
import random
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class FaceForensics(Dataset):
    def __init__(self, opt, split='train', transforms=None, downsample=1, balance=False, resize_shape=(512, 512)):
        self.root_dir = '/data0/lfz/data/ACA'
        self.split = split
        self.transforms = transforms
        self.downsample = downsample
        self.balance = balance
        self.resize_shape = resize_shape  # 新增属性，用于指定统一输入尺寸
        
        # 指定训练或测试路径
        self.data_path = os.path.join(self.root_dir, 'processed' if split == 'train' else 'processed_test')

        # 加载真实视频和伪造视频的数据项
        self.real_items = self._load_items(label=0)  # label0为真实视频
        self.fake_items = self._load_items(label=1)  # label1到label4为伪造视频
        
        # 平衡正负样本
        pos_len = len(self.real_items)
        neg_len = len(self.fake_items)

        if self.split == 'train' and balance:
            self.real_items, self.fake_items = self._balance_data(self.real_items, self.fake_items, pos_len, neg_len)

        # 合并数据项
        self.items = self.real_items + self.fake_items
        self.items = sorted(self.items, key=lambda x: x['img_path'])

        print(f'Total number of data: {len(self.items)} | real: {len(self.real_items)}, fake: {len(self.fake_items)}')
    
    def _load_items(self, label):
        """读取label0-label5下的所有视频文件夹及其图片"""
        items = []
        label_dir = os.path.join(self.data_path, f'label{label}')
        
        for video_folder in os.listdir(label_dir):
            video_path = os.path.join(label_dir, video_folder, 'images')
            if os.path.isdir(video_path):
                imgs = sorted(os.listdir(video_path))  # 排序以保证图片顺序一致
                for i in range(0, len(imgs), self.downsample):
                    img_path = os.path.join(video_path, imgs[i])
                    items.append({'img_path': img_path, 'label': label})
        return items

    def _balance_data(self, real_items, fake_items, pos_len, neg_len):
        """平衡正负样本数量"""
        if pos_len > neg_len:
            real_items = np.random.choice(real_items, neg_len, replace=False).tolist()
        else:
            fake_items = np.random.choice(fake_items, pos_len, replace=False).tolist()
        return real_items, fake_items

    def __getitem__(self, index):
        """返回图片、标签和掩码"""
        item = self.items[index]
        label = item['label']
        image = cv2.imread(item['img_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if label == 0 and image.shape[0] == 1080:
            image = image[100:980, :]  # Crop the top and bottom 100 pixels
        # 调整图像大小以统一输入尺寸
        image = cv2.resize(image, self.resize_shape)

        if self.transforms:
            image = self.transforms[self.split](image=image)['image']

        return image, label

    def __len__(self):
        return len(self.items)
