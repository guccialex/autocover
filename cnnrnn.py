import torch
from torchvision import transforms
from PIL import Image
import random
import os

print( torch.cuda.is_available() )


print("Loading data")

imagedimensions = (64,64)


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



def add_score_and_imagetensor_to_array(path, array):

    randomletter = random.choice("abcdefghijklmnopqrstuvwxyz")
    randomletter += random.choice("abcdefghijklmnopqrstuvwxyz")
    

    count = 0
    for filename in os.listdir(path):

        listname = randomletter + filename

        # count += 1
        # if count > 100000:
        #     break

        if filename.endswith("noncomm.jpg"):
            array.append([listname, image_loader(path + filename), torch.FloatTensor([-1])])
        elif filename.endswith("comm.jpg"):
            array.append([listname, image_loader(path + filename), torch.FloatTensor([1])])
        else:
            continue



trainingdata = []

add_score_and_imagetensor_to_array("mainstreams/zec/zec4/", trainingdata)
# add_score_and_imagetensor_to_array("mainstreams/zec/zec1/", trainingdata)
# add_score_and_imagetensor_to_array("mainstreams/zec/zec2/", trainingdata)
# add_score_and_imagetensor_to_array("mainstreams/zec/zec3/", trainingdata)


#sort by filename
trainingdata.sort(key=lambda x: x[0])

sequencedata = []

inputstack = []
labelstack = []

#split training data into an array of 100 tensors
for (name, input, label) in trainingdata:
    inputstack.append(input)
    labelstack.append(label)

    if len(inputstack) == 200:
        sequencedata.append([inputstack, labelstack])
        
        inputstack = []
        labelstack = []


# trainingdata = newtrainingdata

#shuffle the training data
random.shuffle(sequencedata)

#trainingdataset = torch.utils.data.TensorDataset(torch.stack([x[1] for x in trainingdata]), torch.stack([x[2] for x in trainingdata]))
# trainingdataloader = torch.utils.data.DataLoader(trainingdataset, batch_size=100,  num_workers=2)

# print( len(trainingdata) )





#Define a LSTM that takes in a sequence of Tensors and the hidden state and outputs a single float at every step
class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias=True, batch_first=True):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first=batch_first)
        self.lfc1 = torch.nn.Linear(hidden_size, 70)
        self.lfc2 = torch.nn.Linear(70, 1)
        torch.nn.Linear

        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 13 * 13, 400)
        self.fc2 = torch.nn.Linear(400, 200)
        self.fc3 = torch.nn.Linear(200, input_size)
    
    def forward(self, x, hidden=None):

        #shape of [10, 3, 64, 64]

        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 13 * 13)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        x = x.unsqueeze(0)


        out, hidden = self.lstm(x, hidden)

        out = torch.nn.functional.relu(self.lfc1(out))
        out = self.lfc2(out)
        
        return out, hidden


lstm = LSTM(120, 120, 2)


criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.0001 )

#torch.autograd.set_detect_anomaly(True)

#input = trainingdata[0][1]


average = 0.0
counter = 0

#create a stack of 10 inputs

for epoch in range(10):



    for (inputstack, labelstack) in sequencedata:

        input = torch.stack(inputstack)
        labels = torch.stack(labelstack)
        labels = labels.unsqueeze(0)
        #input with shape [10, 3, 64, 64]

        output, _ = lstm(input, None)

        print(output.shape)

        lstm.zero_grad()

        loss = criterion(output, labels)

        print( loss.shape )

        average += loss.item()

        loss.backward()

        optimizer.step()

        counter += 1
        if counter % 10 == 0:
            print("loss: " + str(average / 10))
            average = 0.0




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
