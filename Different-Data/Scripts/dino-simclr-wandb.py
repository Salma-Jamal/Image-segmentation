import os
import glob
import time
import random
import re
from collections import defaultdict
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
from torchvision.transforms import (
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ColorJitter,
    RandomGrayscale,
    RandomApply,
    Compose,
    GaussianBlur,
    ToTensor,
    Normalize,
    CenterCrop,
    Resize
)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageOps
import seaborn as sns
import pandas as pd
import cv2

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve
from sklearn.decomposition import PCA

# Import wandb
import wandb

# Set device
torch.cuda.set_device("cuda:0")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'DEVICE: {DEVICE}')


# In[4]:


run_name = 'simclr_rad-dino_pos-pairs_aug-pairs_100_epoch_m&ms_4'



class Config:
    def __init__(self):
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.batch_size = 70  # Adjust as needed
        self.patience = 30
        self.dropout_p = 0.3
        self.image_shape = [256, 256]
        self.kernel_size = [21, 21]  # For the transforms, 10% of image size
        self.embedding_size = 128
        self.scheduler_step_size = 70
        self.scheduler_gamma = 0.1
        self.weight_decay = 1e-5
        self.max_norm = 1.0  # Gradient clipping
        self.temperature = 2.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_path = f"/home/saahmed/scratch/projects/Image-segmentation/Different-Data/rerieval_checkpoint/{run_name}"
        os.makedirs(self.base_path, exist_ok=True)
        self.best_model_path = os.path.join(self.base_path, "best_model.pth")
        self.last_model_path = os.path.join(self.base_path, "last_model.pth")
        self.learning_plot_path = os.path.join(self.base_path, "learning_curves.png")

config = Config()


# In[6]:


os.environ["WANDB_API_KEY"] = "4f8dccbaced16f201316dd4113139739694dfd3b"


# In[7]:


# Initialize wandb and log the configuration parameters.
wandb.init(
    project="simclr-training",
    name=run_name,
    config={
        "learning_rate": config.learning_rate,
        "num_epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "dropout_p": config.dropout_p,
        "image_shape": config.image_shape,
        "embedding_size": config.embedding_size,
        "scheduler_step_size": config.scheduler_step_size,
        "scheduler_gamma": config.scheduler_gamma,
        "weight_decay": config.weight_decay,
        "max_norm": config.max_norm,
        "temperature": config.temperature,
    }
)

# Optionally, add the config to wandb for reference
wandb_config = wandb.config



def convert_to_rgb(img):
    return img.convert("RGB")

class AugmentationSequenceType(Enum):
    temp = "temp"
    normal = "normal"

augmentation_sequence_map = {
    AugmentationSequenceType.temp.value: transforms.Compose([
        transforms.Resize((config.image_shape[0], config.image_shape[1])),
        transforms.Lambda(convert_to_rgb),
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Lambda(lambda img: transforms.functional.adjust_contrast(img, contrast_factor=random.uniform(1, 1.3))),
        transforms.ToTensor(),
    ]),
    AugmentationSequenceType.normal.value: transforms.Compose([
        transforms.Resize((config.image_shape[0], config.image_shape[1])),
        transforms.Lambda(convert_to_rgb),
        transforms.ToTensor(),
    ]),
}

class ContrastiveLearningViewGenerator(object):
    def __init__(self, base_transform, normal_transform, n_views=2):
        self.base_transform = base_transform
        self.normal_transform = normal_transform
        self.n_views = n_views

    def __call__(self, x):
        if random.random() < 0.5:
            views = [self.base_transform(x) for _ in range(self.n_views)]
        else:
            views = [self.normal_transform(x), self.base_transform(x)]
        return views

class CombinedContrastiveDataset(Dataset):
    def __init__(self, list_images, positive_pairs, base_transform, normal_transform):
        self.list_images = list_images
        self.positive_pairs = positive_pairs
        self.all_images = self.positive_pairs + self.list_images
        self.base_transform = base_transform
        self.normal_transform = normal_transform
        self.view_generator = ContrastiveLearningViewGenerator(base_transform, normal_transform, n_views=2)
        
    def __len__(self):
        return len(self.list_images) + len(self.positive_pairs)
    
    def __getitem__(self, idx):
        if idx < len(self.positive_pairs):
            img_path1, img_path2 = self.all_images[idx]
            img1 = Image.open(img_path1)
            img2 = Image.open(img_path2)
            img1 = self.normal_transform(img1) 
            img2 = self.normal_transform(img2)
            return [img1, img2]
        else:
            img_path = self.all_images[idx]
            img = Image.open(img_path)
            views = self.view_generator(img)
            return views

# Prepare training image list
images_list_train = []
for i in ['es', 'ed']:
    path = f'/home/saahmed/scratch/projects/Image-segmentation/datasets/MAndMs/processed_data/Training/{i}/images/'
    images = [os.path.join(path, fname) for fname in os.listdir(path)]
    images_list_train += images

def train_val_test_split(list_filenames, train_size=0.7):
    list_filenames_train, list_filenames_val = train_test_split(
        list_filenames,
        train_size=train_size,
        shuffle=True,
        random_state=42)
    return list_filenames_train, list_filenames_val

list_images = images_list_train
list_images_train, list_images_val = train_val_test_split(list_images)

print("Total number of images: ", len(list_images))
print("Images in train split: ", len(list_images_train))
print("Images in validation split: ", len(list_images_val))

import os
import re
from collections import defaultdict

def create_positive_pairs(images):
    file_list = sorted(images)
    pattern = re.compile(r'([^/]+)/images/([A-Z0-9]+)_slice_(\d+)\.png')  # Matches ed/es and filename
    groups = defaultdict(list)

    for path in file_list:
        match = pattern.search(path)
        if match:
            phase = match.group(1)          # 'ed' or 'es'
            patient = match.group(2)        # e.g., 'C6J5P1'
            slice_num = int(match.group(3)) # e.g., 11
            key = (patient, phase)
            groups[key].append((slice_num, path))
        else:
            print(f"File {path} does not match the expected pattern.")
    
    positive_pairs = []
    for key, slices in groups.items():
        slices.sort(key=lambda x: x[0])  # Sort by slice number
        for i in range(len(slices) - 1):
            curr_slice, img1 = slices[i]
            next_slice, img2 = slices[i + 1]
            if curr_slice + 1 == next_slice:
                positive_pairs.append((img1, img2))

    return positive_pairs


pos_pairs_train = create_positive_pairs(list_images_train)
pos_pairs_val = create_positive_pairs(list_images_val)

output_shape = config.image_shape 
base_transforms = augmentation_sequence_map[AugmentationSequenceType.temp.value]
normal_transforms = augmentation_sequence_map[AugmentationSequenceType.normal.value]

image_ds_train = CombinedContrastiveDataset(
    list_images=list_images_train,
    positive_pairs=pos_pairs_train,
    base_transform=base_transforms,
    normal_transform=normal_transforms)

image_ds_val = CombinedContrastiveDataset(
    list_images=list_images_val,
    positive_pairs=pos_pairs_val,
    base_transform=base_transforms,
    normal_transform=normal_transforms)

BATCH_SIZE = config.batch_size

train_loader = DataLoader(
    image_ds_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

val_loader = DataLoader(
    image_ds_val,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

print("Batches in TRAIN: ", len(train_loader))
print("Batches in VAL: ", len(val_loader))
# Note: If you have a test_loader, print it similarly.


# In[10]:


print("samples in TRAIN: ", len(image_ds_train))
print("samples in VAL: ", len(image_ds_val))


# In[11]:


from transformers import Dinov2Model

class SimCLR(nn.Module):
    def __init__(self, dropout_p=0.5, embedding_size=128, freeze=False, linear_eval=False):
        super().__init__()
        self.linear_eval = linear_eval
        self.dropout_p = dropout_p
        self.embedding_size = embedding_size

        # Load the DINOv2 model (you can change to any pretrained model)
        self.encoder = Dinov2Model.from_pretrained('microsoft/rad-dino')
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(768, 256),  # Assuming DINOv2 has an embedding dimension of 768
            nn.Dropout(p=self.dropout_p),
            nn.ReLU(),
            nn.Linear(256, embedding_size)
        )

    def forward(self, x):
        if not self.linear_eval:
            x = torch.cat(x, dim=0)  # Concatenate the two views
        outputs = self.encoder(x)
        encoding = outputs.last_hidden_state[:, 0]  # Extract the [CLS] token representation
        projection = self.projection(encoding)
        return projection

def save_model(model, save_path):
    model.encoder.save_pretrained(save_path)
    torch.save(model.projection.state_dict(), os.path.join(save_path, 'projection_head.pth'))

def load_model(model_class, load_path, device):
    encoder = Dinov2Model.from_pretrained(load_path)
    model = model_class()
    model.encoder = encoder
    projection_head_path = os.path.join(load_path, 'projection_head.pth')
    model.projection.load_state_dict(torch.load(projection_head_path, map_location=device))
    return model

def plot_training(train_loss_history, save_path, val_loss_history=None):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Train Loss')
    if val_loss_history is not None:
        plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.show()

def contrastrive_loss(features, config):
    """NT-Xent (Normalized Temperature-Scaled Cross Entropy) Loss,
    aka. Contrastive Loss, used in the SimCLR paper.

    IMPORTANT NOTE: We don't really return the loss, but the logits
    and the (synthetic) labels to compute it with CrossEntropyLoss!

    The main idea behind SimCLR and contrastive learning is to learn
    representations that are close for positive pairs and far for negative pairs.
    In the case of SimCLR, a positive pair is two different augmentations
    of the same image, and a negative pair is two augmentations
    of two different images.

    How NT-Xent works:
    - Compute the cosine similarity between the representations
    of all pairs of images in the batch.
    - Apply a softmax to these similarities, but treat the similarity
    of each image with its positive pair as the correct class.
    This means that for each image, the goal is to make the
    softmax probability of its positive pair as high as possible,
    and the softmax probabilities of its negative pairs as low as possible.
    - Compute the cross entropy between these softmax probabilities
    and the true labels (which have a 1 for the positive pair
    and 0 for the negative pairs).
    - The temperature parameter scales the similarities before the softmax.
    A lower temperature makes the softmax output more peaky
    (i.e., the highest value will be much higher than the others,
    and the lower values will be closer to zero),
    while a higher temperature makes the softmax output more uniform.

    Args:
        projections: cat(z1, z2)
        z1: The projection of the first branch/view
        z2: The projeciton of the second branch/view

    Returns:
        the NTxent loss

    Notes on the shapes:
        inputs to model (views): [(B, C, W, H), (B, C, W, H)]
            B: batch size
            C: channels
            W: width
            H: height
            E: embedding size
        outputs from model (projections): [2*B, E]
        LABELS: [2*B, 2*B]
        features = outputs from model: [2*B, E]
        mask: [2*B, 2*B]
        similarity_matrix: [2*B, 2*B-1]
        positives: [2*B, 1]
        negatives: [2*B, 2*B-2]
        logits: [2*B, 2*B-1]
        labels: [2*B]
    """
    # FIXME: Refactor: take config out and pass necessary params, remove capital variables, etc.
    # FIXME: convert into class
    BATCH_SIZE = config.batch_size
    DEVICE = config.device
    TEMPERATURE = config.temperature

    LABELS = torch.cat([torch.arange(BATCH_SIZE) for i in range(2)], dim=0)
    LABELS = (LABELS.unsqueeze(0) == LABELS.unsqueeze(1)).float() # Creates a one-hot with broadcasting
    LABELS = LABELS.to(DEVICE) # 2*B, 2*B

    similarity_matrix = torch.matmul(features, features.T) # 2*B, 2*B
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(LABELS.shape[0], dtype=torch.bool).to(DEVICE)
    # ~mask is the negative of the mask
    # the view is required to bring the matrix back to shape
    labels = LABELS[~mask].view(LABELS.shape[0], -1) # 2*B, 2*B-1
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1) # 2*B, 2*B-1

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1) # 2*B, 1

    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1) # 2*B, 2*B-2

    logits = torch.cat([positives, negatives], dim=1) # 2*B, 2*B-1
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(DEVICE)

    logits = logits / TEMPERATURE

    return logits, labels


# In[12]:


model = SimCLR(dropout_p=config.dropout_p, embedding_size=config.embedding_size).to(config.device)
criterion = nn.CrossEntropyLoss().to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)



wandb.watch(model, log="all")


# In[15]:


def validate(model, val_loader, criterion, config):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for views in val_loader:
            projections = model([view.to(config.device) for view in views])
            logits, labels = contrastrive_loss(projections, config)
            loss = criterion(logits, labels)
            val_loss += loss.item()
    return val_loss / len(val_loader)

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, config, output_freq=2, debug=False):
    model = model.to(config.device)
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    no_improve_epochs = 0
    total_batches = len(train_loader)
    print_every = total_batches // output_freq

    for epoch in tqdm(range(config.num_epochs)):
        start_time = time.time()
        train_loss = 0.0
        model.train()

        for i, views in enumerate(train_loader):
            projections = model([view.to(config.device) for view in views])
            logits, labels = contrastrive_loss(projections, config)
            if debug and (torch.isnan(logits).any() or torch.isinf(logits).any()):
                print("[WARNING]: large logits")
                logits = logits.clamp(min=-10, max=10)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)
        train_loss_history.append(train_loss)

        val_loss = validate(model, val_loader, criterion, config)
        val_loss_history.append(val_loss)

        epoch_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
            "epoch_time": epoch_time
        })

        print(f"Epoch: {epoch+1}, Loss: {train_loss}, Val Loss: {val_loss}, Time: {epoch_time:.2f}s, LR: {current_lr}")

        # Save the last model checkpoint locally and log it as an artifact if needed.
        save_model(model, config.last_model_path)
        # artifact = wandb.Artifact("last-model", type="model", metadata={"epoch": epoch+1})
        # artifact.add_dir(config.last_model_path)
        # wandb.log_artifact(artifact, aliases=["latest"])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, config.best_model_path)

            # # Create a wandb artifact and log the best model checkpoint
            # artifact = wandb.Artifact("best-model", type="model", metadata={"epoch": epoch+1})
            # artifact.add_dir(config.best_model_path)
            # wandb.log_artifact(artifact, aliases=["latest"])
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= config.patience:
                print("Early stopping")
                break

    return train_loss_history, val_loss_history


train_loss_history, val_loss_history = train(model, train_loader, val_loader, criterion, optimizer, scheduler, config)





