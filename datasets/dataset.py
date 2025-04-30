import os
import torch
import torch.utils.data
from PIL import Image
from datasets.data_augment import PairCompose, PairRandomCrop, PairToTensor

class LLdataset:
    def __init__(self, config):
        self.config = config

    def get_loaders(self):
        train_dataset = AllWeatherDataset(os.path.join(self.config.data.data_dir, self.config.data.train_dataset),
                                          patch_size=self.config.data.patch_size)
        val_dataset = AllWeatherDataset(os.path.join(self.config.data.data_dir, self.config.data.val_dataset),
                                        patch_size=self.config.data.patch_size, train=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                                 num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader

class AllWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, train=True):
        super().__init__()

        self.dir = dir
        self.train = train
        self.patch_size = patch_size

        low_dir = os.path.join(dir, 'low')
        high_dir = os.path.join(dir, 'high')

        self.input_names = sorted([os.path.join(low_dir, f) for f in os.listdir(low_dir) if f.endswith('.jpg') or f.endswith('.png')])
        self.gt_names = sorted([os.path.join(high_dir, f) for f in os.listdir(high_dir) if f.endswith('.jpg') or f.endswith('.png')])

        # print(f"Loading dataset from {dir}")
        # print(f"Found {len(self.input_names)} low-quality images in {low_dir}")
        # print(f"Low-quality image files: {self.input_names}")
        # print(f"Found {len(self.gt_names)} high-quality images in {high_dir}")
        # print(f"High-quality image files: {self.gt_names}")

        if self.train:
            self.transforms = PairCompose([
                PairRandomCrop(self.patch_size),
                PairToTensor()
            ])
        else:
            self.transforms = PairCompose([
                PairToTensor()
            ])

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]

        img_id = os.path.basename(input_name).split('.')[0]
        input_img = Image.open(input_name)
        gt_img = Image.open(gt_name)

        input_img, gt_img = self.transforms(input_img, gt_img)

        return torch.cat([input_img, gt_img], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
