from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import os

class VOCDataset(Dataset):
    def __init__(self, rootPath, imagePath, annoPath, transform=None):
        self.rootPath = rootPath
        self.transform = transform
        self.imagePath = imagePath
        self.annoPath = annoPath
        self.files = list(os.listdir(self.rootPath + self.annoPath))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = "/" + self.files[idx]
        # imageName = self.rootPath + self.imagePath + name[:-4] + ".jpg"
        # annoName = self.rootPath + self.annoPath + name
        imageName = self.rootPath + self.imagePath + name
        annoName = self.rootPath + self.annoPath + name[:-4] + ".png"

        # print (imageName)
        # print (annoName)
        image = cv2.imread(imageName, cv2.IMREAD_COLOR)
        annotation = cv2.imread(annoName, cv2.IMREAD_COLOR)
        # try:
        #     print (annotation.shape)
        # except AttributeError:
        #     pass

        if self.transform:
            image = self.transform(image)
            annotation = self.transform(annotation)
        return {"image": image, "annotation": annotation}
