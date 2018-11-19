from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import os

class testDataset(Dataset):
    def __init__(self, rootPath, transform=None):
        self.rootPath = rootPath
        self.transform = transform
        self.files = list(os.listdir(self.rootPath))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = "/" + self.files[idx]
        imageName = self.rootPath + name
        image = cv2.imread(imageName, cv2.IMREAD_COLOR)

        if self.transform:
            image = self.transform(image)
        return {"image": image}
