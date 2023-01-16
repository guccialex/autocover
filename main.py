import torch
from torchvision import transforms
from PIL import Image
import random
import os

print( torch.cuda.is_available() )


# exit()

print("Loading data")

#turn test.png into a tensor
def image_loader(image_name):
    loader = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor()])
    image = Image.open(image_name)
    #remove transparency
    image = image.convert('RGB')
    image = loader(image).float()
    image = torch.autograd.Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    #torch.Size([1, 3, 64, 64])
    return image




#add each image as a tensor to a list for each image in the "comms" folder, and it's score
trainingdata = []
commpath = "mainstreams/zec/zeccomm/"
#these should have a score of 1
for filename in os.listdir(commpath):
    if filename.endswith(".jpg"):
        score = torch.FloatTensor([1])
        trainingdata.append([image_loader( commpath + filename), score])
    else:
        continue
#these should have a score of 0
noncommpath = "mainstreams/zec/zecnoncomm/"
for filename in os.listdir(noncommpath):
    if filename.endswith(".jpg") and random.random() < 0.2:
        score = torch.FloatTensor([-1])
        trainingdata.append([image_loader(noncommpath + filename), score])
    else:
        continue


commpath = "mainstreams/zet/zetcomm/"
#these should have a score of 1
for filename in os.listdir(commpath):
    if filename.endswith(".jpg"):
        score = torch.FloatTensor([1])
        trainingdata.append([image_loader( commpath + filename), score])
    else:
        continue
#these should have a score of 0
noncommpath = "mainstreams/zet/zetnoncomm/"
for filename in os.listdir(noncommpath):
    if filename.endswith(".jpg") and random.random() < 0.2:
        score = torch.FloatTensor([-1])
        trainingdata.append([image_loader(noncommpath + filename), score])
    else:
        continue


# commpath = "mainstreams/zin/zincomm/"
# #these should have a score of 1
# for filename in os.listdir(commpath):
#     if filename.endswith(".jpg"):
#         score = torch.FloatTensor([1])
#         trainingdata.append([image_loader( commpath + filename), score])
#     else:
#         continue

# #these should have a score of 0
# noncommpath = "mainstreams/zin/zinnoncomm/"
# for filename in os.listdir(noncommpath):
#     if filename.endswith(".jpg") and random.random() < 0.2:
#         score = torch.FloatTensor([-1])
#         trainingdata.append([image_loader(noncommpath + filename), score])
#     else:
#         continue


#shuffle the data
random.shuffle(trainingdata)

#split the data into training and testing
training = trainingdata
#training = trainingdata[:int(len(trainingdata) * 0.8)]
#testing = trainingdata[int(len(trainingdata) * 0.8):]





commtesting = []

commpath = "mainstreams/zin/zincomm/"
#these should have a score of 1
for filename in os.listdir(commpath):
    if filename.endswith(".jpg"):
        score = torch.FloatTensor([1])
        commtesting.append([image_loader( commpath + filename), score])
    else:
        continue


noncommtesting = []

#these should have a score of 0
noncommpath = "mainstreams/zin/zinnoncomm/"
for filename in os.listdir(noncommpath):
    if filename.endswith(".jpg") and random.random() < 0.2:
        score = torch.FloatTensor([-1])
        noncommtesting.append([image_loader(noncommpath + filename), score])
    else:
        continue


print("Loaded data")






#define the CNN with a single float output
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 13 * 13, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 13 * 13)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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
