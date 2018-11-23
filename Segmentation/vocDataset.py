from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import os
import xml.etree.ElementTree as ET


class VOCDataset(Dataset):
    def __init__(self, root_path, image_path, segment_path, anno_path=None, transform=None, segment=True):
        self.root_path = root_path
        self.transform = transform
        self.segment = segment
        self.image_path = image_path
        self.segment_path = segment_path
        self.anno_path = anno_path
        self.files = list(os.listdir(self.root_path + self.segment_path))
        # print (sorted(self.files, key=sortFunction))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = "/" + self.files[idx]
        image_name = self.root_path + self.image_path + name[:-4] + ".jpg"
        if self.segment:
            anno_name = self.root_path + self.segment_path + name
        else:
            anno_name = self.root_path + self.anno_path + name[:-4] + ".xml"
        # image_name = self.root_path + self.image_path + name
        # anno_name = self.root_path + self.segment_path + name[:-4] + ".png"

        # print (image_name)
        # print (anno_name)
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        if self.segment:
            annotation = cv2.imread(anno_name, cv2.IMREAD_COLOR)
        else:
            annotation = []
            tree = ET.parse(anno_name)
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
