import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalSiamesNetworkSmall(nn.Module):
    def __init__(self, embedding_dim=3):
        super(ConvolutionalSiamesNetwork, self).__init__()
        
        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=11,stride=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, stride=2),
            
            nn.Conv2d(192, 512, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(512, 768, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=1),

            nn.Conv2d(768, 1024, kernel_size=3,stride=1),
            nn.ReLU(inplace=True)
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, embedding_dim)
        )
        
    def forward(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        embedding = self.fc1(output)
        embedding = F.normalize(embedding, p=2.0, dim=1)
        return embedding



class ConvolutionalSiamesNetwork(nn.Module):
    def __init__(self, embedding_dim=3):
        super(ConvolutionalSiamesNetworkSmall, self).__init__()
        
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3,stride=1, padding=1),#128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),#64
            
            nn.Conv2d(128, 194, kernel_size=3, stride=2, padding=1),#32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),#16

            nn.Conv2d(194, 194, kernel_size=1, stride=1),#16
            nn.ReLU(inplace=True),

            nn.Conv2d(194, 388, kernel_size=3, stride=1, padding=1),#16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),#8

            nn.Conv2d(388, 612, kernel_size=3,stride=2, padding=1),#4
            nn.ReLU(inplace=True),

            nn.Conv2d(612, 786, kernel_size=2,stride=1),#3
            nn.ReLU(inplace=True),

            nn.Conv2d(786, 1024, kernel_size=3,stride=1),#1
            nn.ReLU(inplace=True),
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, embedding_dim)
        )
        
    def forward(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        embedding = self.fc1(output)
        embedding = F.normalize(embedding, p=2.0, dim=1)
        return embedding
