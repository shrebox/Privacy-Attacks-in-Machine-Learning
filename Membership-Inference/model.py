import torch
import torch.nn as nn
from torchvision import models


#Target/Shadow CNN model
class ConvNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, num_classes):
        super(ConvNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        layers = []
        layers.append(nn.Conv2d(in_channels=self.input_dim, out_channels=self.hidden_layers[0], kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(self.hidden_layers[0]))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.ReLU(inplace=True))
        for i in range(1, len(hidden_layers)-1):
            layers.append(nn.Conv2d(in_channels=self.hidden_layers[i-1], out_channels=self.hidden_layers[i], kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(self.hidden_layers[i]))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(self.hidden_layers[-1], self.num_classes))
        self.conv_layer = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.conv_layer(x)
        return out

#Target pretrained Model
class VggModel(nn.Module):
    def __init__(self, num_classes,layer_config,pretrained=True):
        super(VggModel, self).__init__()
        #Load the pretrained VGG11_BN model
        self.pretrained = pretrained
        if self.pretrained:
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
        else: # Baseline VGG11_BN model without IMAGENET weights
            self.vgg_scratch = models.vgg11_bn(pretrained=pretrained)

    def forward(self, x):
        if self.pretrained:
            x = self.model_features(x)
            x = x.squeeze()
            out = self.model_classifier(x)
        else:
            out = self.vgg_scratch(x)
        return out

class AttackMLP(nn.Module):
    # Attack Model
        def __init__(self, input_size, hidden_size=64):
            super(AttackMLP, self).__init__()
            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_size, 2)
            self.softmax = nn.Softmax(dim=1)
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            output = self.softmax(output)
            return output