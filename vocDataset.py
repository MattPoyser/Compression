from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import os
import xml.etree.ElementTree as ET

class VOCDataset(Dataset):
    def __init__(self, rootPath, imagePath, segmentPath, annoPath=None, transform=None, segment=True):
        self.rootPath = rootPath
        self.transform = transform
        self.segment = segment
        self.imagePath = imagePath
        self.segmentPath = segmentPath
        self.annoPath = annoPath
        self.files = list(os.listdir(self.rootPath + self.segmentPath))
        # print (sorted(self.files, key=sortFunction))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = "/" + self.files[idx]
        imageName = self.rootPath + self.imagePath + name[:-4] + ".jpg"
        if self.segment:
            annoName = self.rootPath + self.segmentPath + name
        else:
            annoName = self.rootPath + self.annoPath + name[:-4] + ".xml"
        # imageName = self.rootPath + self.imagePath + name
        # annoName = self.rootPath + self.segmentPath + name[:-4] + ".png"

        # print (imageName)
        # print (annoName)
        image = cv2.imread(imageName, cv2.IMREAD_COLOR)
        if self.segment:
            annotation = cv2.imread(annoName, cv2.IMREAD_COLOR)
        else:
            annotation = []
            tree = ET.parse(annoName)
            root = tree.getroot()
            for object in root.findall('object'):
                name = object.find('name').text
                annotation.append(name)

        # try:
        #     print (annotation.shape)
        # except AttributeError:
        #     pass

        if self.transform:
            image = self.transform(image)
            if self.segment:
                annotation = self.transform(annotation)
            else:
                # annotation = np.array(annotation)
                annotation = annotation[0]
        return {"image": image, "annotation": annotation}

def sortFunction(value):
    return int(value[:-4])