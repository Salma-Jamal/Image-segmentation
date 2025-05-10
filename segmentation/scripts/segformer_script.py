import os
import re
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import zipfile
import platform
import warnings
from glob import glob
from dataclasses import dataclass
 
# To filter UserWarning.
warnings.filterwarnings("ignore", category=UserWarning)
 
import cv2
import requests
import numpy as np
# from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import os

 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
 
 
# For data augmentation and preprocessing.
import albumentations as A
from albumentations.pytorch import ToTensorV2
 
# Imports required SegFormer classes
from transformers import SegformerForSemanticSegmentation
 
# Importing lighting along with a built-in callback it provides.
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint,TQDMProgressBar
 
# Importing torchmetrics modular and functional implementations.
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassF1Score
# To print model summary.
from torchinfo import summary
import wandb
 
# Sets the internal precision of float32 matrix multiplications.
torch.set_float32_matmul_precision('high')
 
# To enable determinism.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
 
# # To render the matplotlib figure in the notebook.
# %matplotlib inline


wandb.login(key="4f8dccbaced16f201316dd4113139739694dfd3b") 


wandb.init(project="Segmentation",
          name="Segformer-b5")

path_ed = '/home/saahmed/scratch/projects/Image-segmentation/datasets/ACDC/processed_data/Training/ed/images/'
path_es = '/home/saahmed/scratch/projects/Image-segmentation/datasets/ACDC/processed_data/Training/es/images/'
ed_img = [path_ed+i for i in os.listdir(path_ed)]
es_img = [path_es+i for i in os.listdir(path_es)]


path_ed = '/home/saahmed/scratch/projects/Image-segmentation/datasets/ACDC/processed_data/Training/ed/masks/'
path_es = '/home/saahmed/scratch/projects/Image-segmentation/datasets/ACDC/processed_data/Training/es/masks/'
ed_mask = [path_ed+i for i in os.listdir(path_ed)]
es_mask = [path_es+i for i in os.listdir(path_es)]

image_paths = sorted(ed_img) + sorted(es_img)
mask_paths = sorted(ed_mask) + sorted(es_mask)

path_ed_test = '/home/saahmed/scratch/projects/Image-segmentation/datasets/ACDC/processed_data/Testing/ed/images/'
path_es_test = '/home/saahmed/scratch/projects/Image-segmentation/datasets/ACDC/processed_data/Testing/es/images/'
ed_img_test = [path_ed_test+i for i in os.listdir(path_ed_test)]
es_img_test = [path_es_test+i for i in os.listdir(path_es_test)]

path_ed_test = '/home/saahmed/scratch/projects/Image-segmentation/datasets/ACDC/processed_data/Testing/ed/masks/'
path_es_test = '/home/saahmed/scratch/projects/Image-segmentation/datasets/ACDC/processed_data/Testing/es/masks/'
ed_mask_test = [path_ed_test+i for i in os.listdir(path_ed_test)]
es_mask_test = [path_es_test+i for i in os.listdir(path_es_test)]

image_paths_test = sorted(ed_img_test) + sorted(es_img_test)
mask_paths_test = sorted(ed_mask_test) + sorted(es_mask_test)

def calculate_mean_std(image_paths):

    # Initialize sums and squared sums for 3 channels (R, G, B)
    channel_sum = np.zeros(3)
    channel_sum_squared = np.zeros(3)
    pixel_count = 0

    for path in tqdm(image_paths):
        img = Image.open(path).convert('RGB')  # Ensure image is in RGB format
        img = np.array(img).astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
        
        # Reshape to (H*W, C) where C = 3 for RGB
        pixels = img.reshape(-1, 3)
        channel_sum += pixels.sum(axis=0)
        channel_sum_squared += (pixels ** 2).sum(axis=0)
        pixel_count += pixels.shape[0]

    # Calculate mean for each channel
    mean = channel_sum / pixel_count
    
    # Calculate std for each channel using the formula std = sqrt(E[x^2] - (E[x])^2)
    std = np.sqrt(channel_sum_squared / pixel_count - mean ** 2)
    
    return mean, std

mean, std = calculate_mean_std(image_paths)
print("Mean per channel:", mean)
print("Std per channel:", std)



import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

def is_mask_empty(mask):
    """
    Check if a mask is empty.
    Here, empty means all pixel values are zero.
    """
    return np.all(mask == 255)


valid_images = []
valid_masks = []

for img_path, mask_path in zip(image_paths,mask_paths):
    if not os.path.exists(mask_path):
        print(f"Mask file not found for image: {img_path}")
        continue
    
    # Read mask image as grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Could not read mask file: {mask_path}")
        continue
    
    # If the mask is empty, skip this image
    if is_mask_empty(mask):
        # print(f"{mask_path}\n\n")
        # print(np.unique(mask))
        continue
    
    valid_images.append(img_path)
    valid_masks.append(mask_path)

print("Total valid image-mask pairs:", len(valid_images))

# Split the valid image-mask pairs into training and validation sets
train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    valid_images, valid_masks, test_size=0.2, random_state=42
)

print("Number of training pairs:", len(train_imgs))
print("Number of validation pairs:", len(val_imgs))


test_images = []
test_masks = []

for img_path, mask_path in zip(image_paths_test,mask_paths_test):
    if not os.path.exists(mask_path):
        print(f"Mask file not found for image: {img_path}")
        continue
    
    # Read mask image as grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Could not read mask file: {mask_path}")
        continue
    
    # If the mask is empty, skip this image
    if is_mask_empty(mask):
        # print(f"{mask_path}\n\n")
        # print(np.unique(mask))
        continue
    
    test_images.append(img_path)
    test_masks.append(mask_path)

print("Total valid image-mask pairs:", len(test_images))

@dataclass(frozen=True)
class DatasetConfig:
    NUM_CLASSES:   int = 4 # including background.
    IMAGE_SIZE: tuple[int,int] = (256, 256) # W, H
    MEAN: tuple = (mean[0].item() , mean[1].item() , mean[2].item())
    STD:  tuple = (std[0].item() , std[1].item() , std[2].item())
    BACKGROUND_CLS_ID: int = 0
    DATASET_PATH: str = '/home/saahmed/scratch/projects/Image-segmentation/datasets/ACDC/processed_data/'


@dataclass(frozen=True)
class Paths:
    DATA_TRAIN_IMAGES = train_imgs
    DATA_TRAIN_LABELS = train_masks

    DATA_VALID_IMAGES = val_imgs
    DATA_VALID_LABELS = val_masks

    
    DATA_TEST_IMAGES = test_images
    DATA_TEST_LABELS = test_masks
         
@dataclass
class TrainingConfig:
    BATCH_SIZE:      int = 48 # 32. On colab you should be able to use batch size of 32 with T4 GPU.
    NUM_EPOCHS:      int = 100
    INIT_LR:       float = 3e-4
    NUM_WORKERS:     int = 0 if platform.system() == "Windows" else 12 #os.cpu_count()
 
    OPTIMIZER_NAME:  str = "AdamW"
    WEIGHT_DECAY:  float = 1e-4
    USE_SCHEDULER:  bool = True # Use learning rate scheduler?
    SCHEDULER:       str = "MultiStepLR" # Name of the scheduler to use.
    MODEL_NAME:      str = "nvidia/segformer-b5-finetuned-ade-640-640"
     
 
@dataclass
class InferenceConfig:
    BATCH_SIZE:  int = 10
    NUM_BATCHES: int = 2



# Create a mapping from class ID to RGB color value. Required for visualization.
# (e.g., 0 for background, 85 for right ventricle, 170 for myocardium, and 255 for left ventricle).
id2color = {
    0: (0, 0, 0),    # background pixel (black)
    170: (0, 255, 0),  # Green
    255: (0, 0, 255),  # BLUE
    85: (255, 0, 0),  # RED
}
 
print("Number of classes", DatasetConfig.NUM_CLASSES)
 
# Reverse id2color mapping.
# Used for converting RGB mask to a single channel (grayscale) representation.
rev_id2color = {value: key for key, value in id2color.items()}



class MedicalDataset(Dataset):
    def __init__(self, *, image_paths, mask_paths, img_size, ds_mean, ds_std, is_train=False):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths  
        self.is_train    = is_train
        self.img_size    = img_size
        self.ds_mean = ds_mean
        self.ds_std = ds_std
        self.transforms  = self.setup_transforms(mean=self.ds_mean, std=self.ds_std)
 
    def __len__(self):
        return len(self.image_paths)
 
    def setup_transforms(self, *, mean, std):
        transforms = []
 
        # Augmentation to be applied to the training set.
        if self.is_train:
            transforms.extend([
                A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(scale_limit=0.12, rotate_limit=0.15, shift_limit=0.12, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.CoarseDropout(max_holes=8, max_height=self.img_size[1]//20, max_width=self.img_size[0]//20, min_holes=5, fill_value=0, mask_fill_value=0, p=0.5)
            ])
 
        # Preprocess transforms - Normalization and converting to PyTorch tensor format (HWC --> CHW).
        transforms.extend([
                A.Normalize(mean=mean, std=std, always_apply=True),
                ToTensorV2(),  # (H, W, C) --> (C, H, W)
        ])
        return A.Compose(transforms)
 
    def load_file(self, file_path, depth=0):
        file = cv2.imread(file_path, depth)
        if depth == cv2.IMREAD_COLOR:
            file = file[:, :, ::-1]
        return cv2.resize(file, (self.img_size), interpolation=cv2.INTER_NEAREST)
 
    def __getitem__(self, index):
        # Load image and mask file.
        image = self.load_file(self.image_paths[index], depth=cv2.IMREAD_COLOR)
        mask  = self.load_file(self.mask_paths[index],  depth=cv2.IMREAD_GRAYSCALE)
        # Apply Preprocessing (+ Augmentations) transformations to image-mask pair
        transformed = self.transforms(image=image, mask=mask)
        image, mask = transformed["image"], transformed["mask"].to(torch.long)
        return image, mask


class MedicalSegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        num_classes=10,
        img_size=(384, 384),
        ds_mean=(0.485, 0.456, 0.406),
        ds_std=(0.229, 0.224, 0.225),
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        shuffle_validation=False,
    ):
        super().__init__()
 
        self.num_classes = num_classes
        self.img_size    = img_size
        self.ds_mean     = ds_mean
        self.ds_std      = ds_std
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.pin_memory  = pin_memory
         
        self.shuffle_validation = shuffle_validation

    def setup(self, *args, **kwargs):
        # # Create training dataset and dataloader.

        train_imgs = Paths.DATA_TRAIN_IMAGES
        train_msks = Paths.DATA_TRAIN_LABELS
        
        valid_imgs =  Paths.DATA_VALID_IMAGES
        valid_msks = Paths.DATA_VALID_LABELS

        test_imgs =  Paths.DATA_TEST_IMAGES
        test_msks = Paths.DATA_TEST_LABELS

 
        self.train_ds = MedicalDataset(image_paths=train_imgs, mask_paths=train_msks, img_size=self.img_size,  
                                       is_train=True, ds_mean=self.ds_mean, ds_std=self.ds_std)
 
        self.valid_ds = MedicalDataset(image_paths=valid_imgs, mask_paths=valid_msks, img_size=self.img_size, 
                                       is_train=False, ds_mean=self.ds_mean, ds_std=self.ds_std)

        self.test_ds = MedicalDataset(image_paths=test_imgs, mask_paths=test_msks, img_size=self.img_size, 
                                       is_train=False, ds_mean=self.ds_mean, ds_std=self.ds_std)
 
    def train_dataloader(self):
        # Create train dataloader object with drop_last flag set to True.
        return DataLoader(
            self.train_ds, batch_size=self.batch_size,  pin_memory=self.pin_memory, 
            num_workers=self.num_workers, drop_last=True, shuffle=True
        )    
 
    def val_dataloader(self):
        # Create validation dataloader object.
        return DataLoader(
            self.valid_ds, batch_size=self.batch_size,  pin_memory=self.pin_memory, 
            num_workers=self.num_workers, shuffle=self.shuffle_validation
        )


    def test_dataloader(self):
        # Create validation dataloader object.
        return DataLoader(
            self.test_ds, batch_size=self.batch_size,  pin_memory=self.pin_memory, 
            num_workers=self.num_workers, shuffle=self.shuffle_validation
        )




dm = MedicalSegmentationDataModule(
    num_classes=DatasetConfig.NUM_CLASSES,
    img_size=DatasetConfig.IMAGE_SIZE,
    ds_mean=DatasetConfig.MEAN,
    ds_std=DatasetConfig.STD,
    batch_size=InferenceConfig.BATCH_SIZE,
    num_workers=0,
    shuffle_validation=True,
)
 

# Create training & validation dataset.
dm.setup()
 
train_loader, valid_loader = dm.train_dataloader(), dm.val_dataloader()



def num_to_rgb(num_arr, color_map=id2color):
    single_layer = np.squeeze(num_arr)
    output = np.zeros(num_arr.shape[:2] + (3,))
 
    for k in color_map.keys():
        output[single_layer == k] = color_map[k]
 
    # return a floating point array in range [0.0, 1.0]
    return np.float32(output) / 255.0


def image_overlay(image, segmented_image):
    alpha = 1.0  # Transparency for the original image.
    beta = 0.7  # Transparency for the segmentation map.
    gamma = 0.0  # Scalar added to each sum.
 
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
    image = cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
    return np.clip(image, 0.0, 1.0)



def display_image_and_mask(*, images, masks, color_map=id2color):
    title = ["GT Image", "Color Mask", "Overlayed Mask"]
 
    for idx in range(images.shape[0]):
        image = images[idx]
        grayscale_gt_mask = masks[idx]
 
        fig = plt.figure(figsize=(15, 4))
 
        # Create RGB segmentation map from grayscale segmentation map.
        rgb_gt_mask = num_to_rgb(grayscale_gt_mask, color_map=color_map)
 
        # Create the overlayed image.
        overlayed_image = image_overlay(image, rgb_gt_mask)
 
        plt.subplot(1, 3, 1)
        plt.title(title[0])
        plt.imshow(image)
        plt.axis("off")
 
        plt.subplot(1, 3, 2)
        plt.title(title[1])
        plt.imshow(rgb_gt_mask)
        plt.axis("off")
 
        plt.imshow(rgb_gt_mask)
        plt.subplot(1, 3, 3)
        plt.title(title[2])
        plt.imshow(overlayed_image)
        plt.axis("off")
 
        plt.tight_layout()
        plt.show()
 
    return


def denormalize(tensors, *, mean, std):
    for c in range(3):
        tensors[:, c, :, :].mul_(std[c]).add_(mean[c])
 
    return torch.clamp(tensors, min=0.0, max=1.0)


def get_model(*, model_name, num_classes):
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    return model


model = get_model(model_name=TrainingConfig.MODEL_NAME, num_classes=DatasetConfig.NUM_CLASSES)



summary(model, input_size=(1, 3, *DatasetConfig.IMAGE_SIZE[::-1]), depth=2, device="cpu")


def dice_coef_loss(predictions, ground_truths, num_classes=2, dims=(1, 2), smooth=1e-8):
    """Smooth Dice coefficient + Cross-entropy loss function."""
 
    ground_truth_oh = F.one_hot(ground_truths, num_classes=num_classes)
    prediction_norm = F.softmax(predictions, dim=1).permute(0, 2, 3, 1)
    intersection = (prediction_norm * ground_truth_oh).sum(dim=dims)
    summation = prediction_norm.sum(dim=dims) + ground_truth_oh.sum(dim=dims)
 
    dice = (2.0 * intersection + smooth) / (summation + smooth)
    dice_mean = dice.mean()
 
    CE = F.cross_entropy(predictions, ground_truths)
 
    return (1.0 - dice_mean) + CE



import torch

def remap_labels(target, mapping):
    """
    Remap the labels in the target tensor according to the provided mapping.
    
    Args:
        target (torch.Tensor): The original target tensor with labels to be remapped.
        mapping (dict): A dictionary mapping original labels to new labels.
    
    Returns:
        torch.Tensor: The target tensor with remapped labels.
    """
    remapped_target = torch.zeros_like(target, dtype=torch.long)
    for original_label, new_label in mapping.items():
        remapped_target[target == original_label] = new_label
    return remapped_target



class MedicalSegmentationModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int = 10,
        init_lr: float = 0.001,
        optimizer_name: str = "Adam",
        weight_decay: float = 1e-4,
        use_scheduler: bool = False,
        scheduler_name: str = "multistep_lr",
        num_epochs: int = 100,
    ):
        super().__init__()
 
        # Save the arguments as hyperparameters.
        self.save_hyperparameters()
 
        # Loading model using the function defined above.
        self.model = get_model(model_name=self.hparams.model_name, num_classes=self.hparams.num_classes)
        
        # Define the label mapping
        self.label_mapping = {0: 0, 85: 1, 170: 2, 255: 3}
        
        # Initializing the required metric objects.
        self.mean_train_loss = MeanMetric()
        self.mean_train_f1 = MulticlassF1Score(num_classes=self.hparams.num_classes, average="macro")
        self.mean_valid_loss = MeanMetric()
        self.mean_valid_f1 = MulticlassF1Score(num_classes=self.hparams.num_classes, average="macro")
 
    def forward(self, data):
        outputs = self.model(pixel_values=data, return_dict=True)
        upsampled_logits = F.interpolate(outputs["logits"], size=data.shape[-2:], mode="bilinear", align_corners=False)
        return upsampled_logits
     
    def training_step(self, batch, *args, **kwargs):
        data, target = batch
        
        target = remap_labels(target, self.label_mapping)
        
        logits = self(data)
 
        # Calculate Combo loss (Segmentation specific loss (Dice) + cross entropy)
        loss = dice_coef_loss(logits, target, num_classes=self.hparams.num_classes)
         
        self.mean_train_loss(loss, weight=data.shape[0])
        self.mean_train_f1(logits.detach(), target)
 
        self.log("train/batch_loss", self.mean_train_loss, prog_bar=True, logger=False)
        self.log("train/batch_f1", self.mean_train_f1, prog_bar=True, logger=False)
        # print(f"loss= {loss},  mean_train_loss = {self.mean_train_loss},  mean_train_f1= {self.mean_train_f1}")
        return loss
 
    def on_train_epoch_end(self):
        # Compute the mean training loss and F1 score for the epoch
        train_loss = self.mean_train_loss.compute()
        train_f1 = self.mean_train_f1.compute()
        
        # Log the metrics for display in the progress bar and logger
        self.log("train/loss", train_loss, prog_bar=True)
        self.log("train/f1", train_f1, prog_bar=True)
        self.log("epoch", self.current_epoch)
        
        # Print the metrics to the console
        print(f"Epoch {self.current_epoch}: Train Loss: {train_loss}, Train F1: {train_f1}")
        
        # Reset the metrics for the next epoch
        self.mean_train_loss.reset()
        self.mean_train_f1.reset()
    
    def validation_step(self, batch, *args, **kwargs):
        data, target = batch
        target = remap_labels(target, self.label_mapping)
        logits = self(data)
         
        # Calculate Combo loss (Segmentation specific loss (Dice) + cross entropy)
        loss = dice_coef_loss(logits, target, num_classes=self.hparams.num_classes)
 
        self.mean_valid_loss.update(loss, weight=data.shape[0])
        self.mean_valid_f1.update(logits, target)
 
    def on_validation_epoch_end(self):
        # Compute the mean validation loss and F1 score for the epoch
        valid_loss = self.mean_valid_loss.compute()
        valid_f1 = self.mean_valid_f1.compute()
        
        # Log the metrics for display in the progress bar and logger
        self.log("valid/loss", valid_loss, prog_bar=True)
        self.log("valid/f1", valid_f1, prog_bar=True)
        self.log("epoch", self.current_epoch)
        
        # Print the metrics to the console
        print(f"Epoch {self.current_epoch}: Valid Loss: {valid_loss}, Valid F1: {valid_f1}")
        
        # Reset the metrics for the next epoch
        self.mean_valid_loss.reset()
        self.mean_valid_f1.reset()
     
    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer_name)(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hparams.init_lr,
            weight_decay=self.hparams.weight_decay,
        )
 
        LR = self.hparams.init_lr
        WD = self.hparams.weight_decay
 
        if self.hparams.optimizer_name in ("AdamW", "Adam"):
            optimizer = getattr(torch.optim, self.hparams.optimizer_name)(model.parameters(), lr=LR, 
                                                                          weight_decay=WD, amsgrad=True)
        else:
            optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=WD)
 
        if self.hparams.use_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.trainer.max_epochs // 2,], gamma=0.1)
 
            # The lr_scheduler_config is a dictionary that contains the scheduler
            # and its associated configuration.
            lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "epoch", "name": "multi_step_lr"}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
 
        else:
            return optimizer


 # Seed everything for reproducibility.
pl.seed_everything(42, workers=True)
 
# Intialize custom model.
model = MedicalSegmentationModel(
    model_name=TrainingConfig.MODEL_NAME,
    num_classes=DatasetConfig.NUM_CLASSES,
    init_lr=TrainingConfig.INIT_LR,
    optimizer_name=TrainingConfig.OPTIMIZER_NAME,
    weight_decay=TrainingConfig.WEIGHT_DECAY,
    use_scheduler=TrainingConfig.USE_SCHEDULER,
    scheduler_name=TrainingConfig.SCHEDULER,
    num_epochs=TrainingConfig.NUM_EPOCHS,
) 
 
# Initialize custom data module.
data_module = MedicalSegmentationDataModule(
    num_classes=DatasetConfig.NUM_CLASSES,
    img_size=DatasetConfig.IMAGE_SIZE,
    ds_mean=DatasetConfig.MEAN,
    ds_std=DatasetConfig.STD,
    batch_size=TrainingConfig.BATCH_SIZE,
    num_workers=TrainingConfig.NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# Creating ModelCheckpoint callback. 
# We'll save the model on basis on validation f1-score.
model_checkpoint = ModelCheckpoint(
    monitor="valid/f1",
    mode="max",
    filename="ckpt_{epoch:03d}-vloss_{valid/loss:.4f}_vf1_{valid/f1:.4f}",
    auto_insert_metric_name=False,
)
 
# Creating a learning rate monitor callback which will be plotted/added in the default logger.
lr_rate_monitor = LearningRateMonitor(logging_interval="epoch")


# Initialize logger.
wandb_logger = WandbLogger(log_model=True, project="Segmentation")


# Initializing the Trainer class object.
trainer = pl.Trainer(
    accelerator="auto",  # Auto select the best hardware accelerator available
    devices="auto",  # Auto select available devices for the accelerator (For eg. mutiple GPUs)
    strategy="auto",  # Auto select the distributed training strategy.
    max_epochs=TrainingConfig.NUM_EPOCHS,  # Maximum number of epoch to train for.
    enable_model_summary=False,  # Disable printing of model summary as we are using torchinfo.
    callbacks=[model_checkpoint, lr_rate_monitor],  # Declaring callbacks to use.
    precision="16-mixed",  # Using Mixed Precision training.
    # callbacks=[TQDMProgressBar(refresh_rate=1)]
    logger=wandb_logger
)
 
# Start training
trainer.fit(model, data_module)