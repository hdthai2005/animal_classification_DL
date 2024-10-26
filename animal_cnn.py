import torch
import torch.nn as nn

class ANIMALMODEL(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.conv1 = self._make_block(in_channels= 3, out_channels= 16)
        self.conv2 = self._make_block(in_channels=16, out_channels=32)
        self.conv3 = self._make_block(in_channels=32, out_channels=64)
        self.conv4 = self._make_block(in_channels=64, out_channels=128)
        self.conv5 = self._make_block(in_channels=128, out_channels=128)

        self.fc1 = nn.Sequential(
            nn.Dropout(p= 0.5),
            nn.Linear(in_features=128*7*7, out_features=2048),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    def _make_block(self, in_channels, out_channels, kernel_size = 3):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )

if __name__ == '__main__':
    fake_data = torch.rand(8, 3, 224, 224)
    model = ANIMALMODEL(num_classes = 10)
    prediction = model(fake_data)
    print(prediction.shape)