# -*- coding: utf-8 -*-
"""
Created on Wed Jun 1 15:11:46 2022

@author: Natia_Mestvirishvili
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
import os
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.models.segmentation import fcn_resnet50
from torchvision.utils import make_grid
import torchvision.transforms.functional as F_Transforms
from torchvision.transforms import ConvertImageDtype
import time
import json

class FixationDataset(Dataset):
    def __init__(self, root_dir, image_file, fixation_file, image_transform=None, fixation_transform=None):
        self.root_dir = root_dir
        self.image_files = read_text_file(image_file)
        self.fixation_files = read_text_file(fixation_file)
        self.image_transform = image_transform
        self.fixation_transform = fixation_transform
        assert len(self.image_files) == len(self.fixation_files), "lengths of image files and fixation files do not match!"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = imageio.imread(img_name)

        fix_name = os.path.join(self.root_dir, self.fixation_files[idx])
        fix = imageio.imread(fix_name)

        if self.image_transform:
            image = self.image_transform(image)
        if self.fixation_transform:
            fix = self.fixation_transform(fix)
        
        sample = {"image": image, "fixation": fix, "raw_image": image}
        
        mean, std = image.mean([1,2]), image.std([1,2])
        transform_norm = transforms.Compose([
            transforms.Normalize(mean, std)
        ])
        sample["image"] = transform_norm(image)

        return sample

class FixationTestDataset(Dataset):
    def __init__(self, root_dir, test_file, test_transform=None):
        self.root_dir = root_dir
        self.test_files = read_text_file(test_file)
        self.test_transform = test_transform

    def __len__(self):
        return len(self.test_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.test_files[idx])
        image = imageio.imread(img_name)

        if self.test_transform:
            image = self.test_transform(image)
        
        sample = {"image": image, "raw_image": image, "image_name": img_name}
        
        mean, std = image.mean([1,2]), image.std([1,2])
        transform_norm = transforms.Compose([
            transforms.Normalize(mean, std)
        ])
        sample["image"] = transform_norm(image)

        return sample

class EyeFixationTransform:
    def __init__(self):
        # initialize any properties if necessary
        pass
    def __call__(self, x):
        # do something to get new_x
        new_x = x
        return new_x
        pass

class Eye_Fixation_CNN(nn.Module):
    def __init__(self, resnet_model, center_bias):
        super().__init__()
        self.resnet_model = resnet_model
        self.gauss_kernel = torch.nn.Parameter(data=gaussian_kernel(25, 11.2), requires_grad=False)
        self.center_bias = torch.nn.Parameter(data=torch.log(center_bias), requires_grad=False)
    def forward(self, xb):
        
        xb = F.conv2d(xb, self.gauss_kernel, padding='same')
        xb = self.resnet_model.forward(xb)
        xb = xb['out']+self.center_bias
        
        return xb

def construct_fcn_with_resnet_backbone():
    resnet_model = fcn_resnet50(pretrained=False, pretrained_backbone=True, num_classes=1)
    for param in resnet_model.backbone.parameters():
        param.requires_grad = False
    return resnet_model

def read_text_file(filename):
    lines = []
    with open(filename, 'r') as file:
        for line in file: 
            line = line.strip()
            lines.append(line)
    return lines

def load_data(data_type):
    image_transform = transforms.Compose([transforms.ToTensor(), EyeFixationTransform()])
    fixation_transform = transforms.Compose([transforms.ToTensor(), EyeFixationTransform()])
    paths_dict = load_paths()
    root_dir = paths_dict['project_data']
    train_images_path = paths_dict['train_images_path']
    validation_images_path = paths_dict['validation_images_path']
    train_fixations_path = paths_dict['train_fixations_path']
    validation_fixations_path = paths_dict['validation_fixations_path']
    
    if (data_type == "train"):
        fixation_ds = FixationDataset(root_dir, train_images_path, train_fixations_path, image_transform, fixation_transform)
    elif (data_type == "valid"):
        fixation_ds = FixationDataset(root_dir, validation_images_path, validation_fixations_path, image_transform, fixation_transform)
    
    samples = []
    for sample_index in range(fixation_ds.__len__()):
        samples.append(fixation_ds.__getitem__(sample_index))
        
    fixation_loader = DataLoader(fixation_ds, batch_size=16)
    
    return fixation_loader

def load_test_dataset():
    image_transform = transforms.Compose([transforms.ToTensor(), EyeFixationTransform()])
    paths_dict = load_paths()
    root_dir = paths_dict['project_data']
    test_images_path = paths_dict['test_images_path']
    fixation_ds = FixationTestDataset(root_dir, test_images_path, image_transform)
    fixation_loader = DataLoader(fixation_ds, batch_size=1)
    
    return fixation_loader

def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    device, dtype = None, None
    if isinstance(sigma, torch.Tensor):
        device, dtype = sigma.device, sigma.dtype
    x = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma ** 2))
    return gauss / gauss.sum()

def gaussian_kernel(window_size, sigma):
    g = gaussian(window_size, sigma)
    kernel = torch.matmul(g.unsqueeze(-1), g.unsqueeze(-1).t())
    kernel = kernel.expand(3, 3, 25, 25)
    return kernel

def read_center_bias():
    data = np.load(load_paths()['center_bias'])
    return torch.tensor(data)

def load_paths():
    root_dir = "/Users/Natia_Mestvirishvili/Desktop/UHH/Computer Vision II/course_project"
    project_data = os.path.join(root_dir, 'project_data')
    train_images_path = os.path.join(root_dir, 'project_data/train_images.txt')
    validation_images_path = os.path.join(root_dir, 'project_data/val_images.txt')
    test_images_path = os.path.join(root_dir, 'project_data/test_images.txt')
    train_fixations_path = os.path.join(root_dir, 'project_data/train_fixations.txt')
    validation_fixations_path = os.path.join(root_dir, 'project_data/val_fixations.txt')
    center_bias = os.path.join(root_dir, 'project_data/center_bias_density.npy')
    checkpoints_path = os.path.join(root_dir, 'checkpoints')
    predictions = os.path.join(root_dir, 'predictions')
    logs = os.path.join(root_dir, 'logs')
    paths_dict = {'root_dir': root_dir, 
                  'project_data': project_data,
                  'train_images_path': train_images_path,
                  'validation_images_path': validation_images_path,
                  'test_images_path': test_images_path,
                  'train_fixations_path': train_fixations_path,
                  'validation_fixations_path': validation_fixations_path,
                  'center_bias': center_bias,
                  'checkpoints': checkpoints_path,
                  'predictions': predictions,
                  'logs': logs}
    return paths_dict

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F_Transforms.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    
def visualize_images(inputs, fixations, predictions):
    fixations_grid = make_grid(fixations)
    show(fixations_grid)
    pred_normalized = torch.sigmoid(predictions)
    predictions_grid = make_grid(pred_normalized)
    show(predictions_grid)

def create_log_file_for_current_train():
    paths_dict = load_paths()
    logs_path = paths_dict['logs']
    current_timestamp = round(time.time() * 1000)
    file_path = os.path.join(logs_path, str(current_timestamp) + ".log")
    open(file_path, 'w')
    return file_path

def log_network_performance(file_path, text):
    file_object = open(file_path, 'a')
    file_object.write(text)
    file_object.write('\n')
    file_object.close()
    return 0

def save_network_prediction(pred, image_path):
    pred = torch.squeeze(pred, 0)
    pred = torch.squeeze(pred, 0)
    print(pred.shape)
    out = ConvertImageDtype(torch.uint8)(torch.sigmoid(pred))
    out_np = out.numpy()
    imageio.imwrite(image_path, out_np)

#HyperParameters
learning_rate = 0.1

center_bias = read_center_bias()
train_data_loader = load_data("train")
valid_data_loader = load_data('valid')
resnet_model = construct_fcn_with_resnet_backbone()
eye_fixation_model = Eye_Fixation_CNN(resnet_model, center_bias)
opt = optim.SGD(eye_fixation_model.parameters(), lr=learning_rate)
paths_dict = load_paths()
checkpoints_path = paths_dict['checkpoints']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
eye_fixation_model.to(device)
log_file = create_log_file_for_current_train()

epochs = 20
for epoch in range(epochs):
    log_network_performance(log_file, "Epoch: " + str(epoch))
    eye_fixation_model.train()
    train_loss = 0
    for sample in train_data_loader:
        input_image = sample['image']
        pred = eye_fixation_model(input_image)
        loss =  F.binary_cross_entropy_with_logits(pred, sample["fixation"])
        train_loss += loss
        print(loss)
        #visualize_images(sample['image'], sample['fixation'], pred)
        
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    log_network_performance(log_file, "Training loss: " + str(train_loss / len(train_data_loader)))
    print("Epoch:", epoch, "Training loss:", train_loss / len(train_data_loader))
    
    if (epoch%3 == 0):
        eye_fixation_model.eval()
        with torch.no_grad():
            valid_loss = sum(F.binary_cross_entropy_with_logits(eye_fixation_model(sample['image']), sample['fixation']) for sample in valid_data_loader)
    
        log_network_performance(log_file, "Validation loss: " + str(valid_loss / len(valid_data_loader)))
        print('Epoch:', epoch, 'Validation loss:', valid_loss / len(valid_data_loader))
    
    #save a checkpoint
    file_name = "Fixation_CNN_epoch"+str(epoch)+".pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': eye_fixation_model.state_dict(),
        'optimizer_state_dict': opt.state_dict()
        }, os.path.join(checkpoints_path, file_name))   
    
    log_network_performance(log_file, json.dumps(opt.state_dict()))
    
#Run the testing
test_dataloader = load_test_dataset()
predictions_path = paths_dict['predictions']
for test_sample in test_dataloader:
    test_image = test_sample['image']
    pred = eye_fixation_model(test_image)
    img_num = test_sample['image_name'][0].split('/')[-1].split('.')[0].split('-')[1]
    prediction_image_name = "prediction-" + img_num
    prediction_image_path = os.path.join(predictions_path, prediction_image_name + ".png")
    save_network_prediction(pred, prediction_image_path)
    print("prediction saved")
