import torch
import torch.nn as nn
from torchvision import models

#Below methods to claculate input featurs to the FC layer
#and weight initialization for CNN model is based on the below github repo
#Based on :https://github.com/Lab41/cyphercat/blob/master/Utils/models.py
 
def size_conv(size, kernel, stride=1, padding=0):
    out = int(((size - kernel + 2*padding)/stride) + 1)
    return out
    
    
def size_max_pool(size, kernel, stride=None, padding=0): 
    if stride == None: 
        stride = kernel
    out = int(((size - kernel + 2*padding)/stride) + 1)
    return out

#Calculate in_features for FC layer in Shadow Net
def calc_feat_linear_cifar(size):
    feat = size_conv(size,3,1,1)
    feat = size_max_pool(feat,2,2)
    feat = size_conv(feat,3,1,1)
    out = size_max_pool(feat,2,2)
    return out
    
#Calculate in_features for FC layer in Shadow Net
def calc_feat_linear_mnist(size):
    feat = size_conv(size,5,1)
    feat = size_max_pool(feat,2,2)
    feat = size_conv(feat,5,1)
    out = size_max_pool(feat,2,2)
    return out

#Parameter Initialization
def init_params(m): 
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear): 
        nn.init.xavier_normal_(m.weight.data)
        nn.init.zeros_(m.bias)

#####################################################
# Define Target, Shadow and Attack Model Architecture
#####################################################

#Target Model
class TargetNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, size, out_classes):
        super(TargetNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=hidden_layers[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_layers[0]),
            # nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_layers[0], out_channels=hidden_layers[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_layers[1]),
            # nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        features = calc_feat_linear_cifar(size)
        print('In Features for FC layer in Target Model is : {}'.format( features**2 * hidden_layers[1]))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((features**2 * hidden_layers[1]), hidden_layers[2]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layers[2], out_classes)
        )
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.classifier(out)
        return out
        
    
#Shadow Model mimicking target model architecture, for our implememtation is different than target
class ShadowNet(nn.Module):
    def __init__(self, input_dim, hidden_layers,size,out_classes):
        super(ShadowNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=hidden_layers[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_layers[0]),
            # nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_layers[0], out_channels=hidden_layers[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_layers[1]),
            # nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        features = calc_feat_linear_cifar(size)
        print('In Features for FC layer in Shadow Model is : {}'.format( features**2 * hidden_layers[1]))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((features**2 * hidden_layers[1]), hidden_layers[2]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layers[2], out_classes)
        )
        
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.classifier(out)
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

#Target/Shadow Model for MNIST
class MNISTNet(nn.Module):
    def __init__(self, input_dim, n_hidden,out_classes=10,size=28):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=n_hidden, kernel_size=5),
            nn.BatchNorm2d(n_hidden),
            # nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_hidden, out_channels=n_hidden*2, kernel_size=5),
            nn.BatchNorm2d(n_hidden*2),
            # nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        features = calc_feat_linear_mnist(size)
        print('In Features for FC layer in Shadow Model is : {}'.format(features**2 * (n_hidden*2)))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features**2 * (n_hidden*2), n_hidden*2),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden*2, out_classes)
        )
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.classifier(out)
        return out

#Attack MLP Model
class AttackMLP(nn.Module):
    def __init__(self, input_size, hidden_size=64,out_classes=2):
        super(AttackMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, out_classes)
        )    
    def forward(self, x):
        out = self.classifier(x)
        return out          
