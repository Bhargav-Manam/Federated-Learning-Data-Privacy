
import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

import requests, zipfile, io

from matplotlib import pyplot as plt


class FacesDataset(Dataset):
    """Face dataset."""

    def __init__(self, root_dir, url):
        """
        Args:
            root_dir (string): Directory where the images are saved
            url (string): Url to the remove zip file with images
        """

        raw_data_zip = requests.get(url)
        raw_data = zipfile.ZipFile(io.BytesIO(raw_data_zip.content))
        raw_data.extractall(root_dir)

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, ],
                                 std=[0.229, ])
        ])

        image_folder = os.path.join(root_dir, "faces/training")
        dataset = torchvision.datasets.ImageFolder(image_folder, transform=transform)

        self.data = dataset[0][0]
        self.targets = torch.tensor(dataset[0][1], dtype=torch.int).view(1)
        for idx in range(1, len(dataset)):
            img, lbl = dataset[idx]
            lbl = torch.tensor(lbl, dtype=torch.int).view(1)
            self.data = torch.cat((self.data, img), dim=0)
            self.targets = torch.cat((self.targets, lbl))

    def __len__(self) -> int:
        return len(self.data)
