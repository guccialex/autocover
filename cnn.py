
from torchvision.models import mnasnet0_5, MNASNet0_5_Weights

weights = MNASNet0_5_Weights.DEFAULT

mnasmodel = mnasnet0_5(weights)

mnasmodel.eval()





# cnnmodel = torch.nn.Sequential(*(list(model.children())[:-1]))

# print( list( (list(model.children())[1]).children() ) )

# lastpart = torch.nn.Sequential(*(list(model.children())[1]))

# # exit()

# print(lastpart)

# # #the shape of the output
# output = lastpart(torch.randn(1,1280))
# print(output.shape)

# #flatted
# output = model(torch.randn(1,3,224,224)).view(1,-1).shape
# print(output)

# exit()


# children_counter = 0
# for n,c in cnn.mnasmodel.named_children():
#     print("Children Counter: ",children_counter," Layer Name: ",n,)
#     children_counter+=1

# exit()



# from torchvision.models import resnet18, ResNet18_Weights

# # Initialize model
# weights = ResNet18_Weights.IMAGENET1K_V1
# resnetmodel = resnet18(weights=weights)

# # Set model to eval mode
# resnetmodel.eval()
