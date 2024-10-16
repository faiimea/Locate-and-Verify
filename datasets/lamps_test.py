import os
import cv2
from torch.utils.data import Dataset

class FaceForensics(Dataset):
    def __init__(self, opt, split='train', transforms=None, downsample=1, balance=False, resize_shape=(512, 512)):
        self.root_dir = '/data0/lfz/data/ACA'
        self.split = split
        self.transforms = transforms
        self.downsample = downsample
        self.balance = balance
        self.resize_shape = resize_shape
        
        # Use extracted frames as dataset source
        self.data_path = os.path.join(self.root_dir, 'test_bgr')

        # Load images (all labeled as 1)
        self.items = self._load_items()  # Load all items labeled as 1
        
        print(f'Total data: {len(self.items)}')

    def _load_items(self):
        """Load images from the new folder structure"""
        items = []
        
        for video_folder in os.listdir(self.data_path):
            video_path = os.path.join(self.data_path, video_folder)
            if os.path.isdir(video_path):
                imgs = sorted(os.listdir(video_path))  # Ensure image order
                for i in range(0, len(imgs), self.downsample):
                    img_path = os.path.join(video_path, imgs[i])
                    items.append({'img_path': img_path, 'label': 1})  # All labeled as 1
        return items

    def __getitem__(self, index):
        """Return image and label"""
        item = self.items[index]
        label = item['label']
        image = cv2.imread(item['img_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Crop real video frames if size is 604x1080 (real video size)
        if image.shape == (1080, 604, 3):
            image = image[100:980, :]  # Crop top and bottom 100 pixels

        # Resize image to desired shape
        image = cv2.resize(image, self.resize_shape)

        if self.transforms:
            image = self.transforms[self.split](image=image)['image']

        return image, label

    def __len__(self):
        return len(self.items)
import os
import cv2
from torch.utils.data import Dataset

class FaceForensics(Dataset):
    def __init__(self, opt, split='train', transforms=None, downsample=1, balance=False, resize_shape=(512, 512)):
        self.root_dir = '/data0/lfz/data/ACA'
        self.split = split
        self.transforms = transforms
        self.downsample = downsample
        self.balance = balance
        self.resize_shape = resize_shape
        
        # Use extracted frames as dataset source
        self.data_path = os.path.join(self.root_dir, 'test_bgr')

        # Load images (all labeled as 1)
        self.items = self._load_items()  # Load all items labeled as 1
        
        print(f'Total data: {len(self.items)}')

    def _load_items(self):
        """Load images from the new folder structure"""
        items = []
        
        for video_folder in os.listdir(self.data_path):
            video_path = os.path.join(self.data_path, video_folder)
            if os.path.isdir(video_path):
                imgs = sorted(os.listdir(video_path))  # Ensure image order
                for i in range(0, len(imgs), self.downsample):
                    img_path = os.path.join(video_path, imgs[i])
                    items.append({'img_path': img_path, 'label': 1})  # All labeled as 1
        return items

    def __getitem__(self, index):
        """Return image and label"""
        item = self.items[index]
        label = item['label']
        

        image = cv2.imread(item['img_path'])

        if image is None:
            print(f"Failed to load image: {item['img_path']}")
            raise FileNotFoundError(f"Could not read image: {item['img_path']}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Crop real video frames if size is 604x1080 (real video size)
        if image.shape == (1080, 604, 3):
            image = image[100:980, :]  # Crop top and bottom 100 pixels

        # Resize image to desired shape
        image = cv2.resize(image, self.resize_shape)

        if self.transforms:
            image = self.transforms[self.split](image=image)['image']

        return image, label

    def __len__(self):
        return len(self.items)
