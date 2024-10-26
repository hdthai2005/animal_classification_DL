from torch.utils.data import Dataset, DataLoader
import os
import pickle
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage, Resize, Compose

class AnimalDataset(Dataset):
    def __init__(self, root, is_train, transform = None):
        if is_train:
            data_path = os.path.join(root, "train")
        else:
            data_path = os.path.join(root, "test")
        categories = ['butterfly', 'cat', 'chicken','cow','dog','elephant','horse','sheep','spider','squirrel']
        self.all_image_paths = []
        self.all_labels = []
        self.categories = categories
        for index, category in enumerate(categories):
            category_path = os.path.join(data_path, category)
            for item in os.listdir(category_path):
                image_path = os.path.join(category_path, item)
                self.all_image_paths.append(image_path)
                self.all_labels.append(index)
        self.transform = transform

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, item):
        image_path = self.all_image_paths[item]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.all_labels[item]
        return image, label

if __name__ == '__main__':
    transformer = Compose([
        ToTensor(),
        Resize((224, 224)),
    ])
    dataset = AnimalDataset(root = "animals", is_train=True, transform = transformer)
    image, label = dataset[10]
    print(image, label)