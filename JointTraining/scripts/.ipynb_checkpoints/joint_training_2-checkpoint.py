import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model, SegformerForSemanticSegmentation
from safetensors.torch import load_file
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torchmetrics import MeanMetric
from PIL import Image
import os
import numpy as np
from dataclasses import dataclass
import random
from tqdm import tqdm
import time
import wandb


wandb.login(key="4f8dccbaced16f201316dd4113139739694dfd3b") 


wandb.init(project="joint-training",
          name="joint-training_experiment")

# ----------------------------
# Model Definitions
# ----------------------------

class SimCLR(nn.Module):
    def __init__(self, dropout_p=0.5, embedding_size=128, freeze=False, linear_eval=False):
        super().__init__()
        self.linear_eval = linear_eval
        self.dropout_p = dropout_p
        self.embedding_size = embedding_size
        self.encoder = Dinov2Model.from_pretrained('microsoft/rad-dino')
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(768, 256),
            nn.Dropout(p=self.dropout_p),
            nn.ReLU(),
            nn.Linear(256, embedding_size)
        )

    def forward(self, x):
        if not self.linear_eval:
            x = torch.cat(x, dim=0)
        outputs = self.encoder(x)
        encoding = outputs.last_hidden_state[:, 0]
        projection = self.projection(encoding)
        return projection

retrieval_checkpoint = '/home/saahmed/scratch/projects/Image-segmentation/retrieval/checkpoints/simclr_rad-dino_pos-pairs_aug-pairs_epochs100/best_model/'

retrieval_model = SimCLR(dropout_p=0.3, embedding_size=128, freeze=False, linear_eval=True)
encoder_state_dict = load_file(os.path.join(retrieval_checkpoint, 'model.safetensors'))
retrieval_model.encoder.load_state_dict(encoder_state_dict)
projection_state_dict = torch.load(os.path.join(retrieval_checkpoint, 'projection_head.pth'), map_location=torch.device('cpu'))
retrieval_model.projection.load_state_dict(projection_state_dict)


class JointSegmentationModel(nn.Module):
    def __init__(self, base_model, composite_in_channels=11):
        super().__init__()
        # Map composite input channels to 3 channels expected by segformer
        self.input_adapter = nn.Conv2d(composite_in_channels, 3, kernel_size=1)
        self.base_model = base_model

    def forward(self, x):
        x = self.input_adapter(x)
        outputs = self.base_model(pixel_values=x, return_dict=True)
        # Upsample logits to match input resolution
        logits = F.interpolate(outputs["logits"], size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits

def dice_coef_loss(predictions, ground_truths, num_classes=4, dims=(1, 2), smooth=1e-8):
    ground_truth_oh = F.one_hot(ground_truths, num_classes=num_classes)
    prediction_norm = F.softmax(predictions, dim=1).permute(0, 2, 3, 1)
    intersection = (prediction_norm * ground_truth_oh).sum(dim=dims)
    summation = prediction_norm.sum(dim=dims) + ground_truth_oh.sum(dim=dims)
    dice = (2.0 * intersection + smooth) / (summation + smooth)
    dice_mean = dice.mean()
    CE = F.cross_entropy(predictions, ground_truths)
    return (1.0 - dice_mean) + CE

def differentiable_top2(similarity, tau=1.0):
    weights1 = F.gumbel_softmax(similarity, tau=tau, hard=True)
    B, N = similarity.shape
    masked_similarity = similarity.clone()
    indices = weights1.argmax(dim=-1, keepdim=True)
    for i in range(B):
        masked_similarity[i, indices[i]] = float('-inf')
    weights2 = F.gumbel_softmax(masked_similarity, tau=tau, hard=True)
    return weights1, weights2


class JointTrainingModule(nn.Module):
    def __init__(self, retrieval_model, segmentation_model, num_classes=4, lr=1e-4):
        super().__init__()
        self.retrieval_model = retrieval_model
        self.segmentation_model = segmentation_model
        self.lr = lr
        self.num_classes = num_classes

    def forward(self, query_image, gallery_images, gallery_masks):
        # Compute query embedding
        query_embedding = self.retrieval_model(query_image)
        B, N, C, H, W = gallery_images.shape
        gallery_images_flat = gallery_images.view(B * N, C, H, W)

        gallery_embeddings_list = []
        
        def compute_embedding(x):
            return self.retrieval_model(x)
        
        print('Calculating Embeddings')
        start_time = time.time()
        tempbatch_size = 70
        for i in range(0, B * N, tempbatch_size):
            batch = gallery_images_flat[i:i + tempbatch_size]
            emb = torch_checkpoint(compute_embedding, batch, use_reentrant=False)
            # emb = self.retrieval_model(batch)
            gallery_embeddings_list.append(emb)
        print('Finished Calculating Embeddings')
        
        gallery_embeddings_flat = torch.cat(gallery_embeddings_list, dim=0)
        emb_dim = gallery_embeddings_flat.shape[1]
        gallery_embeddings = gallery_embeddings_flat.view(B, N, emb_dim)
        end_time = time.time()
        print(f"Calculating Embeddings Time: {end_time - start_time:.4f} seconds\n")


        
        print('Calculating similarity')
        start_time = time.time()
        similarity = F.cosine_similarity(query_embedding.unsqueeze(1), gallery_embeddings, dim=-1)
        end_time = time.time()
        print('Finished Calculating similarity')
        print(f"Calculating similarity Time: {end_time - start_time:.4f} seconds\n")
        

        print('Calculating differentiable top2')
        start_time = time.time()
        weights1, weights2 = differentiable_top2(similarity, tau=1.0)
        end_time = time.time()
        print('Finished Calculating differentiable top2')
        print(f"Calculating differentiable top2 Time: {end_time - start_time:.4f} seconds\n")
        

        print('Calculating einsum')
        start_time = time.time()
        guide_image1 = torch.einsum('bn,bnchw->bchw', weights1, gallery_images)
        guide_image2 = torch.einsum('bn,bnchw->bchw', weights2, gallery_images)
        guide_mask1 = torch.einsum('bn,bnhw->bhw', weights1, gallery_masks.half())
        guide_mask2 = torch.einsum('bn,bnhw->bhw', weights2, gallery_masks.half())
        end_time = time.time()
        print('Finished Calculating einsum')
        print(f"Calculating einsum Time: {end_time - start_time:.4f} seconds\n")
        return guide_image1, guide_image2, guide_mask1, guide_mask2


# ----------------------------
# Data Preparation
# ----------------------------

def convert_to_rgb(img):
    return img.convert("RGB")
    
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Lambda(convert_to_rgb),
    transforms.ToTensor(),
])

def remap_labels(target, mapping={0: 0, 85: 1, 170: 2, 255: 3}):
    remapped_target = torch.zeros_like(target, dtype=torch.long)
    for original_label, new_label in mapping.items():
        remapped_target[target == original_label] = new_label
    return remapped_target

# (Paths and file loading logic)
image_dir_ed = '/home/saahmed/scratch/projects/Image-segmentation/datasets/ACDC/processed_data/Training/ed/images/'
mask_dir_ed = '/home/saahmed/scratch/projects/Image-segmentation/datasets/ACDC/processed_data/Training/ed/masks/'
image_dir_es = '/home/saahmed/scratch/projects/Image-segmentation/datasets/ACDC/processed_data/Training/es/images/'
mask_dir_es = '/home/saahmed/scratch/projects/Image-segmentation/datasets/ACDC/processed_data/Training/es/masks/'

image_filenames = sorted([os.path.join(image_dir_ed, i) for i in os.listdir(image_dir_ed)]) + \
                  sorted([os.path.join(image_dir_es, i) for i in os.listdir(image_dir_es)])
mask_filenames = sorted([os.path.join(mask_dir_ed, i) for i in os.listdir(mask_dir_ed)]) + \
                 sorted([os.path.join(mask_dir_es, i) for i in os.listdir(mask_dir_es)])

import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def is_mask_empty(mask):
    return np.all(mask == 255)

valid_images = []
valid_masks = []
for img_path, mask_path in tqdm(zip(image_filenames, mask_filenames), total=len(image_filenames)):
    if not os.path.exists(mask_path):
        continue
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        continue
    if is_mask_empty(mask):
        continue
    valid_images.append(img_path)
    valid_masks.append(mask_path)

print("Total valid image-mask pairs:", len(valid_images))

train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    valid_images, valid_masks, test_size=0.2, random_state=42
)
print("Number of training pairs:", len(train_imgs))
print("Number of validation pairs:", len(val_imgs))

# Prepare guide database (we use a subset for demonstration)
database_images = []
database_masks = []
for img, mask in tqdm(zip(valid_images, valid_masks), total=len(valid_images)):
    g_img = Image.open(img)
    g_mask = Image.open(mask).convert('L')
    processed_img = preprocess(g_img)
    g_mask = g_mask.resize((256,256), resample=Image.NEAREST)
    g_mask = torch.from_numpy(np.array(g_mask)).long()
    g_mask = remap_labels(g_mask)
    database_images.append(processed_img)
    database_masks.append(g_mask)
database_images = torch.stack(database_images, dim=0)
database_masks = torch.stack(database_masks, dim=0)



class JointMedicalDataset(Dataset):
    def __init__(self, image_file_names, mask_file_names, database_images, database_masks, image_size=(256, 256)):
        self.database_images = database_images
        self.database_masks = database_masks
        self.image_filenames = image_file_names
        self.mask_filenames  = mask_file_names
        self.image_size = image_size

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        query_image_path = self.image_filenames[idx]
        query_mask_path = self.mask_filenames[idx]
        query_image = Image.open(query_image_path)
        query_mask = Image.open(query_mask_path).convert('L')
        query_mask = query_mask.resize(self.image_size, resample=Image.NEAREST)
        query_image = preprocess(query_image)
        query_mask = torch.from_numpy(np.array(query_mask)).long()
        query_mask = remap_labels(query_mask)
        return query_image, query_mask, self.database_images, self.database_masks


@dataclass(frozen=True)
class Paths:
    DATA_TRAIN_IMAGES = train_imgs
    DATA_TRAIN_LABELS = train_masks
    DATA_VALID_IMAGES = val_imgs
    DATA_VALID_LABELS = val_masks
    Guide_database_imgs = database_images
    Guide_database_masks = database_masks

# Create datasets and dataloaders
train_ds = JointMedicalDataset(Paths.DATA_TRAIN_IMAGES, Paths.DATA_TRAIN_LABELS,
                               Paths.Guide_database_imgs, Paths.Guide_database_masks)
valid_ds = JointMedicalDataset(Paths.DATA_VALID_IMAGES, Paths.DATA_VALID_LABELS,
                               Paths.Guide_database_imgs, Paths.Guide_database_masks)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_ds, batch_size=4, shuffle=False)


# ----------------------------
# Segmentation Model Setup
# ----------------------------
ckpt_path = '/scratch/saahmed/projects/Image-segmentation/segmentation/lightning_logs/version_0/checkpoints/ckpt_053-vloss_0.1769_vf1_0.9259.ckpt'
checkpoint = torch.load(ckpt_path, map_location="cuda:0")
state_dict = checkpoint["state_dict"]
new_state_dict = {key.replace("model.", ""): value for key, value in state_dict.items()}
config = SegformerForSemanticSegmentation.config_class.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
config.num_labels = 4
segformer_base = SegformerForSemanticSegmentation(config)
segformer_base.load_state_dict(new_state_dict)
segmentation_model = JointSegmentationModel(segformer_base, composite_in_channels=11)



# ----------------------------
# Instantiate Joint Module and Optimizer
# ----------------------------
joint_module = JointTrainingModule(retrieval_model, segmentation_model, num_classes=4, lr=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
joint_module.to(device)
optimizer = torch.optim.Adam(joint_module.parameters(), lr=joint_module.lr)



def train_one_epoch(model, dataloader, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    for batch in tqdm(dataloader, desc="Training"):
        query_image, query_mask, gallery_images, gallery_masks = batch
        # Move tensors to device (no manual half-casting)
        query_image = query_image.to(device)
        query_mask = query_mask.to(device)
        gallery_images = gallery_images.to(device)
        gallery_masks = gallery_masks.to(device)
        
        # Forward pass with autocast
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            # Retrieve guide images/masks
            guide_image1, guide_image2, guide_mask1, guide_mask2 = model(query_image, gallery_images, gallery_masks)
            # Form composite input: query (3) + guide1 (3) + guide2 (3) + guide_mask1 (1) + guide_mask2 (1) = 11 channels
            composite_input = torch.cat([
                query_image,
                guide_image1,
                guide_image2,
                guide_mask1.unsqueeze(1),
                guide_mask2.unsqueeze(1)
            ], dim=1)
            # Forward pass through segmentation model
            print("Calculating Loss")
            start_time = time.time()
            pred_masks = segmentation_model(composite_input)
            # Optionally, if needed you can convert logits to FP32 here:
            # pred_masks = pred_masks.float()
            loss = dice_coef_loss(pred_masks, query_mask, num_classes=model.num_classes)
            end_time = time.time()
            print(f"Calculating Loss Time: {end_time - start_time:.4f} seconds\n")
        
        print("Running optimizer")
        start_time = time.time()
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        end_time = time.time()
        print(f"Running optimizer Time: {end_time - start_time:.4f} seconds\n")
        
        running_loss += loss.item() * composite_input.size(0)
        # Clear cache after each batch
        torch.cuda.empty_cache()
        
        print("#######################################")
        print(f'Running Loss: {running_loss},   loss.item: {loss.item():.4f}')
        wandb.log({"running_loss": running_loss, "loss.item": loss.item()})
        print("#######################################\n")
        
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Epoch Loss: {epoch_loss}")
    wandb.log({"epoch_loss": epoch_loss})
    return epoch_loss

def validate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            query_image, query_mask, gallery_images, gallery_masks = batch
            query_image = query_image.to(device)
            query_mask = query_mask.to(device)
            gallery_images = gallery_images.to(device)
            gallery_masks = gallery_masks.to(device)
            
            with torch.cuda.amp.autocast():
                guide_image1, guide_image2, guide_mask1, guide_mask2 = model(query_image, gallery_images, gallery_masks)
                composite_input = torch.cat([
                    query_image,
                    guide_image1,
                    guide_image2,
                    guide_mask1.unsqueeze(1),
                    guide_mask2.unsqueeze(1)
                ], dim=1)
                pred_masks = segmentation_model(composite_input)
                loss = dice_coef_loss(pred_masks, query_mask, num_classes=model.num_classes)
            running_loss += loss.item() * composite_input.size(0)
            print(f'Running Loss: {running_loss},   loss.item: {loss.item()}')
            wandb.log({"valid_loss.item": loss.item})
            wandb.log({"valid_running_loss": running_loss})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f'Epoch Valid Loss: {epoch_loss}')
    wandb.log({"valid_epoch_loss": epoch_loss})
    return epoch_loss


# ----------------------------
# Run Training
# ----------------------------
best_valid_loss = float("inf")

scaler = torch.amp.GradScaler('cuda', )

num_epochs = 50
for epoch in range(num_epochs):
    train_loss = train_one_epoch(joint_module, train_loader, optimizer, device,scaler)
    valid_loss = validate(joint_module, valid_loader, device)
    print(f"Epoch {epoch+1}/{num_epochs} -- Train Loss: {train_loss:.4f}  Valid Loss: {valid_loss:.4f}")

    # Save checkpoint if validation loss improves
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": joint_module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "valid_loss": valid_loss,
        }
        torch.save(checkpoint, f"best_checkpoint_epoch_{epoch+1}.pth")
        print(f"Checkpoint saved at epoch {epoch+1} with valid loss {valid_loss:.4f}")

