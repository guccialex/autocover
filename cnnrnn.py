import torch
from torchvision import transforms
from PIL import Image
import random
import os


print( torch.cuda.is_available() )


print("Loading data")

imagedimensions = (100,100)

batchsize = 30

numberofimages = 2

#turn test.png into a tensor
def image_loader(image_name):
    loader = transforms.Compose([
        transforms.Resize(imagedimensions[0]),
        transforms.CenterCrop(imagedimensions[0]),
        transforms.ToTensor()])
    image = Image.open(image_name)
    #remove transparency
    image = image.convert('RGB')
    image = loader(image).float()
    image = torch.autograd.Variable(image) #, requires_grad=True)
    # print( image.shape )
    # image = image.unsqueeze(0)
    # print( image.shape )
    #torch.Size([1, 3, 64, 64])
    return image




from torchvision.models import resnet18, ResNet18_Weights

# Initialize model
weights = ResNet18_Weights.IMAGENET1K_V1
resnetmodel = resnet18(weights=weights)

# Set model to eval mode
resnetmodel.eval()


# input = image_loader("test.png")
# input = input.unsqueeze(0)
# print(input.shape)
# output = model.forward(input)
# print(output.shape)
# exit()




def add_score_and_imagetensor_to_array(path, array):

    randomletter = random.choice("abcdefghijklmnopqrstuvwxyz")
    randomletter += random.choice("abcdefghijklmnopqrstuvwxyz")
    randomletter += random.choice("abcdefghijklmnopqrstuvwxyz")
    randomletter += random.choice("abcdefghijklmnopqrstuvwxyz")

    count = 0
    for filename in os.listdir(path):

        listname = randomletter + filename

        # count += 1
        # if count > 100000:
        #     break

        if filename.endswith("noncomm.jpg"):
            array.append([listname, image_loader(path + filename), torch.FloatTensor([0])])
        elif filename.endswith("comm.jpg"):
            array.append([listname, image_loader(path + filename), torch.FloatTensor([1])])
        else:
            continue


def process_data(imagetensor_to_array, databatchsize):

    #sort by filename
    imagetensor_to_array.sort(key=lambda x: x[0])


    inputstack = []
    inputandlabeldata = []

    #split training data into an array of 100 tensors
    for (name, input, label) in imagetensor_to_array:
        inputstack.append(input)

        if len(inputstack) == numberofimages:

            inputandlabeldata.append( (torch.stack(inputstack), label) )
            inputstack = []
    

    #shuffle the data
    random.shuffle(inputandlabeldata)


    inputdata = []
    labeldata = []

    sequencedata = []

    for (input, label) in inputandlabeldata:

        inputdata.append(input)
        labeldata.append(label)

        if len(inputdata) == databatchsize:
            sequencedata.append( (torch.stack(inputdata), torch.stack(labeldata)) )
            inputdata = []
            labeldata = []
    
    return sequencedata



trainingdata = []

#add_score_and_imagetensor_to_array("mainstreams/zec/zec1/", trainingdata)
#add_score_and_imagetensor_to_array("mainstreams/zec/zec2/", trainingdata)
#add_score_and_imagetensor_to_array("mainstreams/zec/zec3/", trainingdata)
add_score_and_imagetensor_to_array("mainstreams/zec/zec4/", trainingdata)


trainingdata = process_data(trainingdata, batchsize)

#shuffle the training data
random.shuffle(trainingdata)


testingdata = []

add_score_and_imagetensor_to_array("mainstreams/zec/zec4/", testingdata)

testingdata = process_data(testingdata, 1)

random.shuffle(testingdata)


#truncate testingdata to 500
testingdata = testingdata[:500]

print("Training data: " + str(len(trainingdata)) )



# inputchannels = 20
# outputchannels = 40

#define a CNN that is a binary classification model
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # self.conv1 = torch.nn.Conv2d(3, inputchannels, 5 , stride=2)
        # self.pool = torch.nn.MaxPool2d(2,2)
        # self.conv2 = torch.nn.Conv2d(inputchannels, outputchannels, 5, stride=2 )
        #numberofimages *
        self.fc1 = torch.nn.Linear(numberofimages * 1000, 3000)
        self.fc2 = torch.nn.Linear(3000, 1500)
        self.fc3 = torch.nn.Linear(1500, 500)
        self.fc4 = torch.nn.Linear(500, 1)

    
    def forward(self, x):

        # print("Start")
        
        x = x.view(-1, 3, imagedimensions[0], imagedimensions[1])

        x = resnetmodel.forward(x)
        
        x = x.view(-1, numberofimages * 1000)

        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))

        return x


cnn = CNN()


# criterion = torch.nn.BCELoss()
# optimizer = torch.optim.SGD(cnn.parameters(), lr=0.0005 )

cnn.load_state_dict(torch.load("cnn2.pt"))

#save as onnx
testinputimage1 = image_loader("test.png")
testinputimage2 = image_loader("test.png")
testinput = torch.stack([testinputimage1, testinputimage2])
testinput = testinput.detach()

print( testinput.shape)


torch.onnx.export(cnn,               # model being run
                testinput,                         # model input (or a tuple for multiple inputs)
                "cnnoutput.onnx",   # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=10,          # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names = ['input'],   # the model's input names
                output_names = ['output'], # the model's output names
                dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                            'output' : {0 : 'batch_size'}})

# torch.onnx.export(cnn,               # model being run
#                 testinput,                         # model input (or a tuple for multiple inputs)
#                 "cnndeep.onnx",   # where to save the model (can be a file or file-like object)
#                 export_params=True,        # store the trained parameter weights inside the model file
#                 opset_version=10,          # the ONNX version to export the model to
#                 do_constant_folding=True,  # whether to execute constant folding for optimization
#                 input_names = ['input'],   # the model's input names
#                 output_names = ['output'], # the model's output names
#                 dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
#                             'output' : {0 : 'batch_size'}})

# #wait 10 seconds
# import time
# time.sleep(10)

exit()

average = 0.0
counter = 0

#create a stack of 10 inputs

for epoch in range(100):
    




    random.shuffle(trainingdata)
    for (inputstack, labelstack) in trainingdata:

        input = inputstack
        labels = labelstack

        output = cnn(input)

        cnn.zero_grad()

        loss = criterion(output, labels)
        
        average += loss.item()

        loss.backward()

        optimizer.step()

        counter += 1
        if counter % len( trainingdata ) // 10 == 0:
            print("loss: " + str(average / 10))
            average = 0.0

            positivecorrect = [0, 0]
            negativecorrect = [0, 0]

            for (inputstack, labelstack) in testingdata:

                input = inputstack
                labels = labelstack

                output = cnn(input)

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
    torch.save(cnn.state_dict(), "cnn2.pt")






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
