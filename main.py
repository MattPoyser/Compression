import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from vocDataset import VOCDataset
from segnetClass import SegNet
import torch.nn as nn
import torchvision
import cv2

def main():
    while(True):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        net = SegNet(3, 45)

        trainset = VOCDataset("asdf", transform)
        trainLoader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)

        train(net, trainLoader)

        # dataiter = iter(trainLoader)
        # keepContinue = True
        # while (keepContinue):
        #     try:
        #         data = dataiter.next()
        #         keepContinue = False
        #         print ("found")
        #     except TypeError:
        #         continue

        # torchvision.utils.save_image(data["image"], "tempImage.jpg")
        # cv2.imshow("images", cv2.imread("tempImage.jpg", cv2.IMREAD_COLOR))
        #
        # torchvision.utils.save_image(data["annotation"], "tempAnno.jpg")
        # cv2.imshow("annotations", cv2.imread("tempAnno.jpg", cv2.IMREAD_COLOR))

        # key = cv2.waitKey(300)
        # if key == ord("x"):
        #     break

    cv2.destroyAllWindows()

def train(net, trainLoader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    #learning rate 0.1 and momentum 0.9 given by segnet paper.
    for epoch in range(2):

        running_loss = 0
        dataiter = iter(trainLoader)
        for i in range(len(trainLoader)):
            try:
                data = dataiter.next()
            except TypeError:
                continue
            inputs = data["image"]
            labels = data["annotation"]

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0



#TODO create segnet architecture cf paper
#TODO train appropriately cf paper
#TODO bayesian?
#TODO save weights
main()