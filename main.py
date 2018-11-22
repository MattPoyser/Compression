import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from vocDataset import VOCDataset
from segnetClass import SegNet
import torch.nn as nn
from testDataset import testDataset
import torchvision
import torch
import cv2
import os
import sys
import numpy as np

def main():
    while(True):
        transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.RandomCrop([256,256], pad_if_needed=True),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = VOCDataset("/home/matt/Documents/segm/VOCdevkit/VOC2007/", "JPEGImages", "SegmentationObject", annoPath="Annotations", transform=transform, segment=True)
        #num_workers = 0 means that the mian process is doing the loading.
        #since each worker loads a batch, then we can have batch_size*num_workers loads before erroring
        #when the data is combined.
        trainLoader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

        savePath = "/home/matt/Documents/segm/weightsFile.txt"
        print (len(sys.argv))
        if os.path.isfile(savePath) and len(sys.argv) == 1:
            net = torch.load(savePath)
        else:
            net = SegNet(3, 45)
            train(net, trainLoader)
            torch.save(net, savePath)

        testset = testDataset("/home/matt/Documents/segm/testPhotos/", transform=transform)
        testLoader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
        test(net, testLoader)

        break

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
    print ("training")
    for epoch in range(2):

        running_loss = 0
        dataiter = iter(trainLoader)
        print (len(trainLoader))
        for i in range(len(trainLoader)): #range(len(trainLoader)) is num. iterations
            #num iterations x batch size = num. images
            try:
                data = dataiter.next()
                print ("success")
            except TypeError as e:
                raise(e)
                # print ("guilty")
                # continue
            inputs = data["image"]
            labels = data["annotation"] #for segmentation, has shape (w,x,y,z) where w is batch size?
            #x is no. channels in image, y,z are (width, height) of image. i.e. label per pixel

            optimizer.zero_grad()

            outputs = net(inputs)
            print (outputs.shape, "guilty")
            print (labels.shape)
            loss = criterion(outputs, labels.long())

            #segment=False
            # loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

#commented lines is for batch_size=1
def test(net, testLoader):
    dataiter = iter(testLoader)
    for i in range(len(testLoader)):
        data = dataiter.next()
        output = net(data)
        # data = dataiter.next()["image"]
        # output = net(data)[0]
        # print (output.shape)

        cv2.imshow("output", torchvision.utils.make_grid(output))
        # torchvision.utils.save_image(output, "tempOutput.jpg")
        # output = cv2.imread("tempOutput.jpg", cv2.IMREAD_COLOR)
        # cv2.imshow("output", output)

        key = cv2.waitKey(300)
        if key == ord('x'):
            break

    cv2.destroyAllWindows()


#TODO train appropriately cf paper
#TODO bayesian?

#TODO batchsize of 1 gives immediate error at max_unpool2d(x4Pool...)
#TODO batchsize of 2 with annoPath images only gives same error as:
#TODO batchsize of 4 with imagePath images. i.e. size of tensors do not match
#TODO since batchsize 2 with annoPath means 211 iterations, which is multiple of 422(length of annoPath)
#TODO then not an issue with last batch having different number of stuff. therefore wat.
#TODO taking out decoder portion of segnet succeeds until test set, but of course generates incorrect output
#TODO as outputs are not same shape as image
main()