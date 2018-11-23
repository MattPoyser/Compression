import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.models import AlexNet
import torch
import sys
import os
import cv2
import numpy as np


classes = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


def main():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    # coco_dataset = torchvision.datasets.CocoDetection(
    #     "/home/matt/Documents/cocoTrain2017/train2017",
    #     "/home/matt/Documents/cocoTrain2017/annotations/person_keypoints_train2017.json",
    #     transform=transform,
    # )

    cifar_dataset = torchvision.datasets.CIFAR10(
        "/home/matt/Documents/",
        train=True,
        transform=transform,
    )

    save_path = "/home/matt/Documents/maskRCNN/weightsFile.txt"
    print(len(sys.argv))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if os.path.isfile(save_path) and len(sys.argv) == 1:
        net = torch.load(save_path)
    else:
        net = AlexNet()
        train_loader = DataLoader(cifar_dataset, batch_size=4, shuffle=True, num_workers=2)
        print(device)
        net.to(device)
        train(train_loader, net, device)
        torch.save(net, save_path)

    test_set = torchvision.datasets.CIFAR10(
        "/home/matt/Documents",
        train=False,
        transform=transform,
    )
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)
    test(test_loader, net, device)


def train(train_loader, net, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

    for epoch in range(2):
        running_loss = 0
        for i, (images, targets) in enumerate(iter(train_loader)):
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = net(images)

            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


def test(test_loader, net, device):
    for i, (images, targets) in enumerate(iter(test_loader)):
        images, targets = images.to(device), targets.to(device)
        outputs = net(images)

        # cv2.imshow("outputs", np.array(torchvision.utils.make_grid(images)))

        grid = torchvision.utils.make_grid(images)
        torchvision.utils.save_image(grid, "tempOutput.jpg")
        output = cv2.imread("tempOutput.jpg", cv2.IMREAD_COLOR)
        cv2.imshow("output", output)

        print([classes[x] for x in targets.cpu().numpy()])

        key = cv2.waitKey(3000)
        if key == ord('x'):
            break

    cv2.destroyAllWindows()


main()
