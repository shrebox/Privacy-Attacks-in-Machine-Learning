import torch
import torch.nn as nn
from torchvision import models

#####################################################
# Define Target, Shadow and Attack Model Architecture
#####################################################

#Target Model CNN based
class TargetNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, num_classes):
        super(TargetNet, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=input_dim, out_channels=hidden_layers[0], kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(hidden_layers[0]))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.ReLU(inplace=True))
        for i in range(1, len(hidden_layers)-1):
            layers.append(nn.Conv2d(in_channels=hidden_layers[i-1], out_channels=hidden_layers[i], kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(hidden_layers[i]))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(512, num_classes))
        self.conv_layer = nn.Sequential(*layers)

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = self.conv_layer(x)
        # print(out.shape)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out
        
        
#Shadow Model CNN based
class ShadowNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, num_classes):
        super(ShadowNet, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=input_dim, out_channels=hidden_layers[0], kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(hidden_layers[0]))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=hidden_layers[0], out_channels=hidden_layers[1], kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(hidden_layers[1]))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=hidden_layers[1], out_channels=hidden_layers[2], kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(hidden_layers[2]))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(2048, hidden_layers[3]))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_layers[3], num_classes))
        self.conv_layer = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.conv_layer(x)
        return out

#Pretrained VGG11 model for Target
class VggModel(nn.Module):
    def __init__(self, num_classes,layer_config,pretrained=True):
        super(VggModel, self).__init__()
        #Load the pretrained VGG11_BN model
        if pretrained:
            pt_vgg = models.vgg11_bn(pretrained=pretrained)

            #Deleting old FC layers from pretrained VGG model
            print('### Deleting Avg pooling and FC Layers ####')
            del pt_vgg.avgpool
            del pt_vgg.classifier

            self.model_features = nn.Sequential(*list(pt_vgg.features.children()))
            
            #Adding new FC layers with BN and RELU for CIFAR10 classification
            self.model_classifier = nn.Sequential(
                nn.Linear(layer_config[0], layer_config[1]),
                nn.BatchNorm1d(layer_config[1]),
                nn.ReLU(inplace=True),
                nn.Linear(layer_config[1], num_classes),
            )

    def forward(self, x):
        x = self.model_features(x)
        x = x.squeeze()
        out = self.model_classifier(x)
        return out

 #Attack MLP Model
class AttackMLP(nn.Module):
   
    def __init__(self, input_size, hidden_size=64,out_classes=2):
        super(AttackMLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc2 = torch.nn.Linear(hidden_size,out_classes)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out          
