import os.path
import torch
from torchvision.transforms import Compose, ToTensor, Resize
import torch.nn as nn
import argparse
from torchvision.models import ResNet18_Weights, resnet18
import cv2
import numpy as np

def get_args():
    parser = argparse.ArgumentParser("Test arguments")
    parser.add_argument("--video_path", "-s", type=str, default = "animals.mp4")
    parser.add_argument("--image_size", "-i", type=int, default=224)
    parser.add_argument("--checkpoint_path", "-t", type=str, default="train_models/animals/best.pt")
    args = parser.parse_args()
    return args


def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    categories = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']
    # ori_image = cv2.imread(args.image_path)
    # image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (args.image_size, args.image_size))
    # image = np.transpose(image, (2, 0, 1))/255.
    # image = np.expand_dims(image, axis = 0)
    # image = torch.from_numpy(image).float()
    # image = image.to(device)

    # model = ANIMALMODEL(num_classes = len(train_dataset.categories)).to(device)
    model = resnet18(weights = None)
    model.fc = nn.Linear(512, 10)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    softmax = nn.Softmax()
    cap = cv2.VideoCapture(args.video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*"MJPG"), int(cap.get(cv2.CAP_PROP_FPS)), (width, height))

    with torch.no_grad():
        while cap.isOpened():
            flag, frame = cap.read()
            if not flag:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (args.image_size, args.image_size))
            image = np.transpose(image, (2, 0, 1))/255.
            image = np.expand_dims(image, axis = 0)
            image = torch.from_numpy(image).float()
            image = image.to(device)

            prediction = model(image)
            prob = softmax(prediction)

            max_value, max_index = torch.max(prob, dim=1)
            cv2.putText(frame,"{}".format(categories[max_index[0]]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
            out.write(frame)

    cap.release()
    out.release()
    # cv2.imshow("The image is about class {} with probality of {}".format(categories[max_index[0]], max_value[0]), ori_image)
    # cv2.waitKey(0)

if __name__ == '__main__':
    args = get_args()
    inference(args)
