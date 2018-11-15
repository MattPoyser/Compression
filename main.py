import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vocDataset import VOCDataset
import torchvision
import torch
import cv2

def main():
    while(True):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = VOCDataset("asdf", transform)
        trainLoader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)

        dataiter = iter(trainLoader)
        keepContinue = True
        while (keepContinue):
            try:
                data = dataiter.next()
                keepContinue = False
                print ("found")
            except TypeError:
                continue

        torchvision.utils.save_image(data["image"], "tempImage.jpg")
        cv2.imshow("images", cv2.imread("tempImage.jpg", cv2.IMREAD_COLOR))

        torchvision.utils.save_image(data["annotation"], "tempAnno.jpg")
        cv2.imshow("annotations", cv2.imread("tempAnno.jpg", cv2.IMREAD_COLOR))

        key = cv2.waitKey(300)
        if key == ord("x"):
            break

    cv2.destroyAllWindows()

main()