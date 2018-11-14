from torch.utils.data import Dataset
import cv2
import os

class VOCDataset(Dataset):
    def __init__(self, rootPath, transform=None):
        self.rootPath = rootPath
        self.rootPath = "/home/matt/Documents/segm/VOCdevkit/VOC2007/"
        self.transform = transform
        self.imagePath = "JPEGImages"
        self.annoPath = "Annotations"
        self.files = list(os.listdir(self.rootPath + self.imagePath))

    def __getlength__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        image = cv2.imread(self.rootPath + self.imagePath + name, cv2.IMREAD_COLOR)
        annotation = cv2.imread(self.rootPath + self.annoPath + name, cv2.IMREAD_COLOR)

        if self.transform:
            image = self.transform(image)

        return {"image": image, "annotation": annotation}

voc = VOCDataset("asdf")
