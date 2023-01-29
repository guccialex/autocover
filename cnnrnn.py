import torch
from torchvision import transforms
from PIL import Image
import random
import os


print( torch.cuda.is_available() )


# x = torch.hub.list('pytorch/vision:v0.14.1', force_reload=True)
# ['alexnet', 'convnext_base', 'convnext_large', 'convnext_small', 'convnext_tiny', 'deeplabv3_mobilenet_v3_large', 'deeplabv3_resnet101', 'deeplabv3_resnet50', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'efficientnet_v2_l', 'efficientnet_v2_m', 'efficientnet_v2_s', 'fcn_resnet101', 'fcn_resnet50', 'get_model_weights', 'get_weight', 'googlenet', 'inception_v3', 'lraspp_mobilenet_v3_large', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'raft_large', 'raft_small', 'regnet_x_16gf', 'regnet_x_1_6gf', 'regnet_x_32gf', 'regnet_x_3_2gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_8gf', 'regnet_y_128gf', 'regnet_y_16gf', 'regnet_y_1_6gf', 'regnet_y_32gf', 'regnet_y_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_8gf', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext101_64x4d', 'resnext50_32x4d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0', 'squeezenet1_1', 'swin_b', 'swin_s', 'swin_t', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'vit_b_16', 'vit_b_32', 'vit_h_14', 'vit_l_16', 'vit_l_32', 'wide_resnet101_2', 'wide_resnet50_2']


import cnn
cnnmodel = cnn.mnasmodel

print("Loading data")

imagedimensions = (256,256)
batchsize = 40
numberofimages = 20


def image_loader(image_name):
    loader = transforms.Compose([
    transforms.Resize((imagedimensions[0], imagedimensions[1])),
    #transforms.CenterCrop(imagedimensions[0]),
    transforms.ToTensor()])
    image = Image.open(image_name)
    #remove transparency
    image = image.convert('RGB')
    image = loader(image).float()
    image = torch.autograd.Variable(image) #, requires_grad=True)
    return image

#turn test.png into a tensor
def image_cnn_loader(image_name):
    image = image_loader(image_name)
    image = torch.unsqueeze(image, 0)
    #image = cnnmodel.forward(image)
    return torch.squeeze(image, 0).detach()








def add_score_and_imagetensor_to_array(path, array):

    count = 0
    for filename in os.listdir(path):

        filepath = path + filename
        
        if filename.endswith("noncomm.jpg"):
            array.append([filepath, image_cnn_loader(filepath), torch.FloatTensor([0])])
        elif filename.endswith("comm.jpg"):
            array.append([filepath, image_cnn_loader(filepath), torch.FloatTensor([1])])
        else:
            continue


def process_data(imagetensor_to_array, databatchsize):

    #sort by filename
    imagetensor_to_array.sort(key=lambda x: x[0])


    tensorstack = []
    tensorandlabeldata = []

    #split training data into an array of 100 tensors
    for (filepath, tensor, label) in imagetensor_to_array:
        tensorstack.append(tensor)

        if len(tensorstack) == numberofimages:

            tensorandlabeldata.append( (torch.stack(tensorstack), label) )
            tensorstack = []
    

    #shuffle the data
    random.shuffle(tensorandlabeldata)


    inputdata = []
    labeldata = []

    sequencedata = []

    for (input, label) in tensorandlabeldata:

        inputdata.append(input)
        labeldata.append(label)

        if len(inputdata) == databatchsize:
            sequencedata.append( (torch.stack(inputdata), torch.stack(labeldata)) )
            inputdata = []
            labeldata = []
    
    return sequencedata



trainingdata = []

add_score_and_imagetensor_to_array("mainstreams/zec/zec1/", trainingdata)
add_score_and_imagetensor_to_array("mainstreams/zec/zec2/", trainingdata)
add_score_and_imagetensor_to_array("mainstreams/zec/zec3/", trainingdata)
#add_score_and_imagetensor_to_array("mainstreams/zec/zec4/", trainingdata)
#add_score_and_imagetensor_to_array("mainstreams/zec/zectest/", trainingdata)

trainingdata = process_data(trainingdata, batchsize)



testingdata = []

#add_score_and_imagetensor_to_array("mainstreams/zec/zec4/", testingdata)
add_score_and_imagetensor_to_array("mainstreams/zec/zectest/", testingdata)

testingdata = process_data(testingdata, 1)


#truncate testingdata to 500
testingdata = testingdata[:500]


print("Training data: " + str(len(trainingdata)) + " testing data: " + str(len(testingdata)))


#define a CNN that is a binary classification model
class CommModel(torch.nn.Module):
    def __init__(self):
        super(CommModel, self).__init__()

        # self.conv1 = torch.nn.Conv2d(3, inputchannels, 5 , stride=2)
        # self.pool = torch.nn.MaxPool2d(2,2)
        # self.conv2 = torch.nn.Conv2d(inputchannels, outputchannels, 5, stride=2 )
        #numberofimages *
        self.fc1 = torch.nn.Linear(numberofimages * 1000, 2000)
        self.fc2 = torch.nn.Linear(2000, 1000)
        self.fc3 = torch.nn.Linear(1000, 500)
        self.fc4 = torch.nn.Linear(500, 1)

    
    def forward(self, x):

        x = x.view(-1, numberofimages * 1000)

        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))

        return x


commmodel = CommModel()


criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(commmodel.parameters(), lr=0.0005 )


exit()



# #save as onnx
# testinput = image_cnn_loader("test.png")
# testinput = torch.unsqueeze(testinput, 0)

# testinput = testinput.detach()

# print( testinput.shape)


# torch.onnx.export(resnetmodel,               # model being run
#                 testinput,                         # model input (or a tuple for multiple inputs)
#                 "models/resnet.onnx",   # where to save the model (can be a file or file-like object)
#                 export_params=True,        # store the trained parameter weights inside the model file
#                 opset_version=10,          # the ONNX version to export the model to
#                 do_constant_folding=True,  # whether to execute constant folding for optimization
#                 input_names = ['input'],   # the model's input names
#                 output_names = ['output'], # the model's output names
#                 dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
#                             'output' : {0 : 'batch_size'}})

# exit()




# testinput = trainingdata[0][0]

# #take the first item in the batch so the testinput goes from size [30, 2, 1000]
# #to size [1, 2, 1000]
# testinput = testinput[0]
# testinput = testinput.unsqueeze(0)

# output = commmodel(testinput)
# print(output)

# print( testinput.shape)


# torch.onnx.export(commmodel,               # model being run
#                 testinput,                         # model input (or a tuple for multiple inputs)
#                 "models/commmodel.onnx",   # where to save the model (can be a file or file-like object)
#                 export_params=True,        # store the trained parameter weights inside the model file
#                 opset_version=10,          # the ONNX version to export the model to
#                 do_constant_folding=True,  # whether to execute constant folding for optimization
#                 input_names = ['input'],   # the model's input names
#                 output_names = ['output'], # the model's output names
#                 dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
#                             'output' : {0 : 'batch_size'}})

# exit()

average = 0.0
counter = 0

#create a stack of 10 inputs

for epoch in range(100):
    
    random.shuffle(trainingdata)
    for (inputstack, labelstack) in trainingdata:

        input = inputstack
        labels = labelstack

        output = commmodel(input)

        commmodel.zero_grad()

        loss = criterion(output, labels)
        
        average += loss.item()

        loss.backward()

        optimizer.step()

        counter += 1
        if counter % len( trainingdata ) // 20 == 0:
            print("loss: " + str(average / 10))
            average = 0.0

            positivecorrect = [0, 0]
            negativecorrect = [0, 0]

            for (inputstack, labelstack) in testingdata:

                input = inputstack
                labels = labelstack

                output = commmodel(input)

                if labels.item() > 0.5:
                    positivecorrect[1] += 1
                    # if output.item() > 0.5:
                    #     positivecorrect[0] += 1
                    positivecorrect[0] += output.item()
                else:
                    negativecorrect[1] += 1
                    # if output.item() < 0.5:
                    #     negativecorrect[0] += 1
                    negativecorrect[0] += output.item()
    

            print("positive accuracy: " + str(positivecorrect) + "  " + str(positivecorrect[0] / positivecorrect[1]))
            print("negative accuracy: " + str(negativecorrect) + "  " + str(negativecorrect[0] / negativecorrect[1]))
    
    print("Epoch: " + str(epoch))
    torch.save(commmodel.state_dict(), "models/commmodel.pt")






exit()




#train the LSTM
for epoch in range(10):

    average = 0.0

    accuracy = 0.0


    for i, data in enumerate(trainingdata, 0):

        hidden = None

        for x in data:


            stackedtensor = torch.stack([data[1], data[2]])
            
            _, inputs, labels = stackedtensor

            lstm.zero_grad()

            outputs, newhidden = lstm(inputs, hidden)

            hidden = newhidden

            loss = criterion(outputs, labels)
            average += loss.item()

            loss.backward(retain_graph=True)
            

            displayevery = 100

            if i % displayevery == displayevery - 1:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, average / displayevery))
                average = 0.0

                optimizer.step()
                hidden = None




#save the model
torch.save(lstm.state_dict(), "lstm.pt")

exit()


#load the model
lstm = LSTM(50, 50, 2)
lstm.load_state_dict(torch.load("lstm.pt"))





#the file to tensor to score
testingdata = []

def add_images_to_testingdata(path, score):

    count = 0
    for filename in os.listdir(path):

        count += 1
        if count > 100000:
            break

        if filename.endswith(".jpg"):
            testingdata.append([filename, image_loader(path + filename), score])
        else:
            continue


add_images_to_testingdata("mainstreams/zet/secondzetcomm/", torch.FloatTensor([1]))
add_images_to_testingdata("mainstreams/zet/secondzetnoncomm/", torch.FloatTensor([-1]))



exit()


net = Net()
criterion = torch.nn.L1Loss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

for epoch in range(10):

    numbereveryprint = 1000

    running_loss = 0.0
    for i, data in enumerate(trainingdata, 0):
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        outputs = outputs.view(-1)

        loss = criterion(outputs, labels)
        loss.backward()

        if i % 50 == 49:
            optimizer.step()

        # print statistics
        running_loss += loss.item()

        
        #every 1000 images, print the loss
        if i % numbereveryprint == numbereveryprint - 1:

            print('[%d, %5d] loss: %.3f' %  (epoch + 1, i + 1, running_loss / numbereveryprint))
            running_loss = 0.0

            #get the average loss for the testing data
            test_loss = 0
            for i, data in enumerate(commtesting, 0):
                inputs, labels = data
                outputs = net(inputs)
                outputs = outputs.view(-1)
                test_loss += outputs.item()
                # loss = criterion(outputs, labels)
                # test_loss += loss.item()
            
            print("comm test loss: " + str(test_loss / len(commtesting)))


                        #get the average loss for the testing data
            test_loss = 0
            for i, data in enumerate(noncommtesting, 0):
                inputs, labels = data
                outputs = net(inputs)
                outputs = outputs.view(-1)
                test_loss += outputs.item()
                # loss = criterion(outputs, labels)
                # test_loss += loss.item()
            
            print("noncomm test loss: " + str(test_loss / len(noncommtesting)))



exit()


#save the model
PATH = './cnn.pth'
torch.save(net.state_dict(), PATH)

#load the model
net = Net()
PATH = './cnn.pth'
net.load_state_dict(torch.load(PATH))




#test the model from "test.png"


testinput = image_loader("/home/lucci/Documents/coding/mnist/test.png")

# Export the model
torch.onnx.export(net,               # model being run
                  testinput,                         # model input (or a tuple for multiple inputs)
                  "cnn.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})



# commaverage = 0
# commtotal = 0


# allfiles = []

# for filename in os.listdir("comms"):
#     if filename.endswith(".png"):
#         path = "comms/" + filename
#         allfiles.append(path)
#     else:
#         continue


# for filename in os.listdir("noncomms"):
#     if filename.endswith(".png"):
#         path = "noncomms/" + filename
#         allfiles.append(path)
#     else:
#         continue

# import shutil

# #delete and then remake the folders commpred and noncommpred
# shutil.rmtree("commpred")
# shutil.rmtree("noncommpred")
# os.mkdir("commpred")
# os.mkdir("noncommpred")


# for path in allfiles:
#     image = image_loader(path)
#     output = net(image)
    
#     if output > 0.7:
#         image = Image.open(path)
#         filename = path.split("/")[-1]
#         image.save("commpred/" + filename)
#     else:
#         image = Image.open(path)
#         filename = path.split("/")[-1]
#         image.save("noncommpred/" + filename)





#read every

# test = image_loader("/home/lucci/Documents/coding/mnist/test.png")
# output = net(test)
# print(output)


# test = image_loader("/home/lucci/Documents/coding/mnist/test2.png")
# output = net(test)
# print(output)

# test = image_loader("/home/lucci/Documents/coding/mnist/test3.png")
# output = net(test)
# print(output)

# test = image_loader("/home/lucci/Documents/coding/mnist/test4.png")
# output = net(test)
# print(output)
