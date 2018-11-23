from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import os


class TestDataset(Dataset):
    def __init__(self, root_path, transform=None):
        self.root_path = root_path
        self.transform = transform
        self.files = list(os.listdir(self.root_path))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = "/" + self.files[idx]
        image_name = self.root_path + name
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)

        if self.transform:
            image = self.transform(image)
        return {"image": image}
