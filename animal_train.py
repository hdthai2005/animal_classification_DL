import os.path
import torch
from animal_dataset import AnimalDataset
from animal_cnn import ANIMALMODEL
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, RandomAffine, ColorJitter
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import argparse
import shutil
from torchvision.models import ResNet18_Weights, resnet18
from torchsummary import summary
def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="twilight")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

def get_args():
    parser = argparse.ArgumentParser("Test arguments")
    parser.add_argument("--image_size", "-i", type=int, default=224)
    parser.add_argument("--batch_size", "-b", type = int, default = 32)
    parser.add_argument("--num_epochs", "-e", type=int, default=100)
    parser.add_argument("--learning_rate", "-l", type=float, default=1e-2)
    parser.add_argument("--momentum", "-m", type=float, default=0.9)
    parser.add_argument("--log_path", "-p", type=str, default="tensorboard/animals")
    parser.add_argument("--data_path", "-d", type=str, default="animals")
    parser.add_argument("--checkpoint_path", "-c", type=str, default="train_models/animals")
    parser.add_argument("--pretrained_checkpoint_path", "-t", type=str, default=None)
    args = parser.parse_args()
    return args

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transformer = Compose([
        ToTensor(),
        RandomAffine(
            degrees = (-5 ,5),
            translate=(0.15, 0.15),
            scale = (0.85, 1.1),
            shear = 10
        ),
        Resize((args.image_size, args.image_size)),
        ColorJitter(
            brightness=0.5,
            contrast=0.8,
            saturation=0.5,
            hue=0.1
        ),
        Normalize(mean = [0.485, 0.456, 0.406],
                  std = [0.229, 0.224, 0.225]),
    ])

    val_transformer = Compose([
        ToTensor(),
        Resize((args.image_size, args.image_size)),
        Normalize(mean = [0.485, 0.456, 0.406],
                  std = [0.229, 0.224, 0.225]),
    ])
    train_dataset = AnimalDataset(root = args.data_path, is_train=True, transform = train_transformer)
    train_params = {
        "batch_size" : args.batch_size,
        "shuffle" : True,
        "num_workers" : 10,
        "drop_last" : True
    }
    train_dataloader = DataLoader(dataset = train_dataset, **train_params)

    val_dataset = AnimalDataset(root = args.data_path, is_train=False, transform = val_transformer)
    val_params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": 10,
        "drop_last": False
    }
    val_dataloader = DataLoader(dataset=val_dataset, **val_params)

    # model = ANIMALMODEL(num_classes = len(train_dataset.categories)).to(device)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 10)
    model.to(device)
    # input(model)
    # summary(model, (3, args.image_size, args.image_size))
    # exit(0)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, momentum=args.momentum)
    if args.pretrained_checkpoint_path:
        checkpoint = torch.load(args.pretrained_checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
    else:
        start_epoch = 0
        best_acc = -1
    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)
    writer = SummaryWriter(args.log_path)

    if not os.path.isdir(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    best_acc = -1
    num_iters_per_epoch = len(train_dataloader)
    for epoch in range(start_epoch, args.num_epochs):
        # MODEL TRAIN
        model.train()
        train_loss = []
        progress_bar = tqdm(train_dataloader, colour = "GREEN")
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            # Forward
            predictions = model(images)
            loss = criterion(predictions, labels)
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            progress_bar.set_description("Epoch : {}/{}. Loss : {:0.4f}".format(epoch + 1, args.num_epochs, np.mean(train_loss)))
            writer.add_scalar("Train/Loss", np.mean(train_loss), epoch * num_iters_per_epoch + iter)

        # MODEL VALIDATION
        all_losses = []
        all_predictions = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            progress_bar = tqdm(val_dataloader, colour = "YELLOW")
            for iter, (images, labels) in enumerate(progress_bar):
                images = images.to(device)
                labels = labels.to(device)
                # Forward pass
                predictions = model(images)
                loss = criterion(predictions, labels)
                predictions = torch.argmax(predictions, 1)
                all_predictions.extend(predictions.tolist())
                all_labels.extend(labels.tolist())
                all_losses.append(loss.item())

        acc = accuracy_score(all_labels, all_predictions)
        loss = np.mean(all_losses)
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        writer.add_scalar("Val/Loss", loss, epoch)
        writer.add_scalar("Val/Accuracy", acc, epoch)
        print("Epoch : {}/{}. Val_Accuracy : {}. Val_Loss : {}".format(epoch + 1, args.num_epochs, acc, loss))
        plot_confusion_matrix(writer, conf_matrix, [i for i in range(10)], epoch)

        check_point = {
            "epoch" : epoch + 1,
            "model" : model.state_dict(),
            "optimizer" : optimizer.state_dict(),
            "best_acc" : best_acc,
        }
        torch.save(check_point, os.path.join(args.checkpoint_path, "last.pt"))
        if acc > best_acc:
            best_acc = acc
            torch.save(check_point, os.path.join(args.checkpoint_path, "best.pt"))
        

if __name__ == '__main__':
    args = get_args()
    train(args)
