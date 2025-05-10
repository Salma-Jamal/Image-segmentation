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


# In[2]:


wandb.login(key="4f8dccbaced16f201316dd4113139739694dfd3b") 


# In[3]:


wandb.init(project="joint-training",
          name="joint-training_experiment-add-segmentation-transforms-only-input-exp")


# In[4]:


# ----------------------------
# Model Definitions
# ----------------------------

class SimCLR(nn.Module):
    def __init__(self, dropout_p=0.5, embedding_size=128, linear_eval=False):
        super().__init__()
        self.linear_eval = linear_eval
        self.dropout_p = dropout_p
        self.embedding_size = embedding_size
        self.encoder = Dinov2Model.from_pretrained('microsoft/rad-dino')
        
        for name, param in self.encoder.named_parameters():
            # Freeze everything by default
            param.requires_grad = False

        for name, param in self.encoder.named_parameters():
            # Unfreeze the last block and the final layernorm:
            if "encoder.layer.11" in name or "layernorm" in name:
                param.requires_grad = True

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

retrieval_model = SimCLR(dropout_p=0.3, embedding_size=128, linear_eval=True)
encoder_state_dict = load_file(os.path.join(retrieval_checkpoint, 'model.safetensors'))
retrieval_model.encoder.load_state_dict(encoder_state_dict)
projection_state_dict = torch.load(os.path.join(retrieval_checkpoint, 'projection_head.pth'), map_location=torch.device('cpu'))
retrieval_model.projection.load_state_dict(projection_state_dict)


# In[5]:


class JointSegmentationModel(nn.Module):
    def __init__(self, base_model, composite_in_channels=7):
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


# In[6]:


def precompute_gallery_embeddings(
    joint_model,
    gallery_images,  # (N, 3, H, W)
    gallery_masks,   # (N, H, W)
    device,
):
    """
    Encodes all gallery images with the retrieval_model once (no extra batch dimension).
    Returns:
        cached_gallery_embeddings: (N, D)  # D = embedding dimension
        cached_gallery_images:     (N, 3, H, W) moved to device
        cached_gallery_masks:      (N, H, W)    moved to device

    NOTE: We wrap this in no_grad() to avoid building a large graph for the gallery.
          That means the gallery side won't backprop on each batch, but the retrieval
          model can still be updated from the query side.
    """
    # Move images/masks to device
    gallery_images = gallery_images.to(device)
    gallery_masks = gallery_masks.to(device)

    N = gallery_images.shape[0]
    gallery_embeddings_list = []

    # print("Calculating Embeddings")
    # start_time = time.time()
    tempbatch_size = 70

    # <-- Turn off gradient tracking for the gallery
    with torch.no_grad():
        for i in range(0, N, tempbatch_size):
            batch = gallery_images[i : i + tempbatch_size]
            # Normal forward pass (no checkpoint, no grad)
            emb = joint_model.retrieval_model(batch)
            gallery_embeddings_list.append(emb)

    # print("Finished Calculating Embeddings")

    # Concatenate chunked embeddings into (N, D)
    cached_gallery_embeddings = torch.cat(gallery_embeddings_list, dim=0)
    # Optionally call .detach() to ensure they have no grad_fn
    cached_gallery_embeddings = cached_gallery_embeddings.detach()

    # end_time = time.time()
    # print(f"Calculating Embeddings Time: {end_time - start_time:.4f} seconds\n")

    return cached_gallery_embeddings, gallery_images, gallery_masks


# In[7]:


class JointTrainingModule(nn.Module):
    def __init__(self, retrieval_model, segmentation_model, num_classes=4, lr=1e-4,tau=0.1):
        super().__init__()
        self.retrieval_model = retrieval_model
        self.segmentation_model = segmentation_model
        self.lr = lr
        self.num_classes = num_classes
        self.tau = tau  # Temperature for the softmax

        self.cached_gallery_embeddings = None  # shape (N, D)
        self.cached_gallery_images = None      # shape (N, 3, H, W)
        self.cached_gallery_masks = None       # shape (N, H, W)

        # self.cached_gallery_embeddings_seg = None  # shape (N, D)
        # self.cached_gallery_images_seg = None      # shape (N, 3, H, W)
        # self.cached_gallery_masks_seg = None       # shape (N, H, W)


    def forward(self, query_image):
        """
        query_image:   (B, 3, H, W)
        gallery_images:(B, N, 3, H, W)
        gallery_masks: (B, N, H, W)
        Returns: guide_image, guide_mask
        """
        # -- 1) Compute query embedding
        query_embedding = self.retrieval_model(query_image)

    
        similarity = F.cosine_similarity(query_embedding.unsqueeze(1), self.cached_gallery_embeddings.unsqueeze(0), dim=-1)

        #     If tau is small, it becomes sharper around the highest similarities.
        weights = F.softmax(similarity / self.tau, dim=-1)  # (B, N)

        # self.cached_gallery_images =  self.cached_gallery_images.cpu()
        # self.cached_gallery_masks =  self.cached_gallery_masks.cpu()

        # self.cached_gallery_images_seg =  self.cached_gallery_images_seg.to(device)
        # self.cached_gallery_masks_seg =  self.cached_gallery_masks_seg.to(device)
        
        # -- 5) Weighted sum to get a single guide_image and guide_mask
        guide_image = torch.einsum("bn,nchw->bchw", weights, self.cached_gallery_images)
        guide_mask = torch.einsum("bn,nhw->bhw", weights, self.cached_gallery_masks.float())
   
        return guide_image, guide_mask


# In[8]:


import albumentations as A
from albumentations.pytorch import ToTensorV2



mean = np.array([0.15452379,0.15452379,0.15452379])
std = np.array([0.17455664 ,0.17455664 ,0.17455664])


def setup_transforms(*, mean, std):
    transforms = []
    # Augmentation to be applied to the training set.
    transforms.extend([
            A.Normalize(mean=mean, std=std, always_apply=True),
            ToTensorV2(),  # (H, W, C) --> (C, H, W)
    ])
    return A.Compose(transforms)

segformer_trans = setup_transforms(mean=mean, std=std)

def read_trans_data_segmentation(image_path,mask_path):

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = image[:, :, ::-1]

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    resized_image = cv2.resize(image,(256, 256), interpolation=cv2.INTER_NEAREST)

    resized_mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
   
    transformed = segformer_trans(image=resized_image, mask=resized_mask)
    image, mask = transformed["image"], transformed["mask"].to(torch.long)

    return image, mask


# In[9]:


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
database_images_seg_trans = []
database_masks_seg_trans = []
for img, mask in tqdm(zip(valid_images, valid_masks), total=len(valid_images)):
    g_img = Image.open(img)
    g_mask = Image.open(mask).convert('L')
    processed_img = preprocess(g_img)
    g_mask = g_mask.resize((256,256), resample=Image.NEAREST)
    g_mask = torch.from_numpy(np.array(g_mask)).long()
    g_mask = remap_labels(g_mask)
    database_images.append(processed_img)
    database_masks.append(g_mask)
    
    img_seg,mask_seg = read_trans_data_segmentation(img,mask)
    database_images_seg_trans.append(img_seg)
    database_masks_seg_trans.append(mask_seg)
    
database_images = torch.stack(database_images, dim=0)
database_masks = torch.stack(database_masks, dim=0)


database_images_seg_trans = torch.stack(database_images_seg_trans, dim=0)
database_masks_seg_trans = torch.stack(database_masks_seg_trans, dim=0)


# In[10]:


db_images = database_images
db_masks = database_masks

db_images_seg = database_images_seg_trans
db_masks_seg = database_masks_seg_trans


# In[11]:


class JointMedicalDataset(Dataset):
    def __init__(self, image_file_names, mask_file_names, image_size=(256, 256)):
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
        
        query_image_seg, query_mask_seg = read_trans_data_segmentation(query_image_path,query_mask_path)
        query_mask_seg = remap_labels(query_mask_seg)
        
        return query_image, query_mask, query_image_seg, query_mask_seg


# In[12]:


# (Paths and file loading logic)
image_dir_ed_Testing = '/home/saahmed/scratch/projects/Image-segmentation/datasets/ACDC/processed_data/Testing/ed/images/'
mask_dir_ed_Testing = '/home/saahmed/scratch/projects/Image-segmentation/datasets/ACDC/processed_data/Testing/ed/masks/'
image_dir_es_Testing = '/home/saahmed/scratch/projects/Image-segmentation/datasets/ACDC/processed_data/Testing/es/images/'
mask_dir_es_Testing = '/home/saahmed/scratch/projects/Image-segmentation/datasets/ACDC/processed_data/Testing/es/masks/'

image_filenames_Testing = sorted([os.path.join(image_dir_ed_Testing, i) for i in os.listdir(image_dir_ed_Testing)]) + \
                  sorted([os.path.join(image_dir_es_Testing, i) for i in os.listdir(image_dir_es_Testing)])
mask_filenames_Testing = sorted([os.path.join(mask_dir_ed_Testing, i) for i in os.listdir(mask_dir_ed_Testing)]) + \
                 sorted([os.path.join(mask_dir_es_Testing, i) for i in os.listdir(mask_dir_es_Testing)])

import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def is_mask_empty(mask):
    return np.all(mask == 255)

valid_images_Testing = []
valid_masks_Testing = []
for img_path, mask_path in tqdm(zip(image_filenames_Testing, mask_filenames_Testing), total=len(image_filenames_Testing)):
    if not os.path.exists(mask_path):
        continue
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        continue
    if is_mask_empty(mask):
        continue
    valid_images_Testing.append(img_path)
    valid_masks_Testing.append(mask_path)

print("Number of Testing pairs:", len(valid_images_Testing))

# Prepare guide database (we use a subset for demonstration)
database_images_Testing = []
database_masks_Testing = []

database_images_Testing_seg_trans = []
database_masks_Testing_seg_trans = []

for img, mask in tqdm(zip(valid_images_Testing, valid_masks_Testing), total=len(valid_images_Testing)):
    g_img = Image.open(img)
    g_mask = Image.open(mask).convert('L')
    processed_img = preprocess(g_img)
    g_mask = g_mask.resize((256,256), resample=Image.NEAREST)
    g_mask = torch.from_numpy(np.array(g_mask)).long()
    g_mask = remap_labels(g_mask)
    database_images_Testing.append(processed_img)
    database_masks_Testing.append(g_mask)
    
    img_seg,mask_seg = read_trans_data_segmentation(img,mask)
    database_images_Testing_seg_trans.append(img_seg)
    database_masks_Testing_seg_trans.append(mask_seg)
    
database_images_Testing = torch.stack(database_images_Testing, dim=0)
database_masks_Testing = torch.stack(database_masks_Testing, dim=0)

database_images_Testing_seg_trans = torch.stack(database_images_Testing_seg_trans, dim=0)
database_masks_Testing_seg_trans = torch.stack(database_masks_Testing_seg_trans, dim=0)


# In[13]:


@dataclass(frozen=True)
class Paths:
    DATA_TRAIN_IMAGES = train_imgs
    DATA_TRAIN_LABELS = train_masks
    
    DATA_VALID_IMAGES = val_imgs
    DATA_VALID_LABELS = val_masks
    
    DATA_TEST_IMAGES = valid_images_Testing
    DATA_TEST_LABELS = valid_masks_Testing
    
    Guide_database_imgs = db_images
    Guide_database_masks = db_masks

    Guide_database_imgs_seg = db_images_seg
    Guide_database_masks_seg = db_masks_seg
    
    Guide_database_imgs_Testing = database_images_Testing
    Guide_database_masks_Testing = database_masks_Testing

    Guide_database_imgs_Testing_seg = database_images_Testing_seg_trans
    Guide_database_masks_Testing_seg = database_masks_Testing_seg_trans

# Create datasets and dataloaders
train_ds = JointMedicalDataset(Paths.DATA_TRAIN_IMAGES, Paths.DATA_TRAIN_LABELS,)
valid_ds = JointMedicalDataset(Paths.DATA_VALID_IMAGES, Paths.DATA_VALID_LABELS,)

test_ds = JointMedicalDataset(Paths.DATA_TEST_IMAGES, Paths.DATA_TEST_LABELS,)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_ds, batch_size=4, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)


# In[14]:


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
segmentation_model = JointSegmentationModel(segformer_base, composite_in_channels=7)


# In[ ]:


# ----------------------------
# Instantiate Joint Module and Optimizer
# ----------------------------
joint_module = JointTrainingModule(retrieval_model, segmentation_model, num_classes=4, lr=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
joint_module.to(device)


# In[16]:


optimizer = torch.optim.Adam(joint_module.parameters(), lr=joint_module.lr)


# In[17]:


def train_one_epoch(model, dataloader, optimizer, device, scaler):
    model.train()
    running_loss = 0.0

    db_images, db_masks =  Paths.Guide_database_imgs, Paths.Guide_database_masks

    # db_images_seg, db_masks_seg =  Paths.Guide_database_imgs_seg, Paths.Guide_database_masks_seg

    cached_gallery_embeddings, cached_gallery_images, cached_gallery_masks = precompute_gallery_embeddings(
        model,
        db_images,        # Tensor of shape (N, 3, H, W)
        db_masks,         # Tensor of shape (N, H, W)
        device,)

    model.cached_gallery_embeddings = cached_gallery_embeddings  # shape (N, D)
    model.cached_gallery_images = cached_gallery_images      # shape (N, 3, H, W)
    model.cached_gallery_masks = cached_gallery_masks       # shape (N, H, W)

    # # model.cached_gallery_embeddings = cached_gallery_embeddings  # shape (N, D)
    # model.cached_gallery_images_seg = db_images_seg     # shape (N, 3, H, W)
    # model.cached_gallery_masks_seg = db_masks_seg     # shape (N, H, W)

    for batch in dataloader:
        query_image, query_mask, query_image_seg , query_mask_seg  = batch
        # Move tensors to device (no manual half-casting)
        query_image = query_image.to(device)
        query_mask = query_mask.to(device)
        
        # Forward pass with autocast
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            
            # Retrieve guide images/masks
            guide_image, guide_mask = model(query_image)

            query_image = query_image.cpu()
            query_image_seg = query_image_seg.to(device)
            # Composite input: 3 (query) + 3 (guide) + 1 (mask) = 7 channels
            composite_input = torch.cat([
                query_image_seg, 
                guide_image, 
                guide_mask.unsqueeze(1)
            ], dim=1)

            
            start_time = time.time()
            pred_masks = model.segmentation_model(composite_input)

            query_mask = query_mask.cpu()
            query_mask_seg = query_mask_seg.to(device)
            
            loss = dice_coef_loss(pred_masks, query_mask_seg, num_classes=model.num_classes)
            end_time = time.time()
            # print(f"Calculating Loss Time: {end_time - start_time:.4f} seconds\n")
        
        # print("Running optimizer")
        start_time = time.time()
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        end_time = time.time()
        # print(f"Running optimizer Time: {end_time - start_time:.4f} seconds\n")
        
        running_loss += loss.item() * composite_input.size(0)
        # Clear cache after each batch
        torch.cuda.empty_cache()

    epoch_loss = running_loss / len(dataloader.dataset)
    # print(f"Train Loss: {epoch_loss}")
    wandb.log({"Train Loss": epoch_loss})
    return epoch_loss

def validate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            query_image, query_mask, query_image_seg , query_mask_seg = batch
            query_image = query_image.to(device)
            query_mask = query_mask.to(device)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # Retrieve guide images/masks
                guide_image, guide_mask = model(query_image)

                query_image = query_image.cpu()
                query_image_seg = query_image_seg.to(device)
                
                # Composite input: 3 (query) + 3 (guide) + 1 (mask) = 7 channels
                composite_input = torch.cat([
                    query_image_seg, 
                    guide_image, 
                    guide_mask.unsqueeze(1)
                ], dim=1)
                
                pred_masks = model.segmentation_model(composite_input)

                query_mask = query_mask.cpu()
                query_mask_seg = query_mask_seg.to(device)
                
                loss = dice_coef_loss(pred_masks, query_mask_seg, num_classes=model.num_classes)
            running_loss += loss.item() * composite_input.size(0)
            # print(f'Running Loss: {running_loss},   loss.item: {loss.item()}')
            # wandb.log({"valid_loss.item": loss.item()})
            # wandb.log({"valid_running_loss": running_loss})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    # print(f'Valid Loss: {epoch_loss}')
    wandb.log({"Valid_Loss": epoch_loss})
    return epoch_loss


# In[18]:


# ----------------------------
# Run Training
# ----------------------------
best_valid_loss = float("inf")

scaler = torch.amp.GradScaler('cuda', )


# Define the folder path where checkpoints will be saved
checkpoint_folder = "/home/saahmed/scratch/projects/Image-segmentation/JointTraining/segmentation-transforms-only-input-exp"

num_epochs = 50
for epoch in tqdm(range(num_epochs)):
    train_loss = train_one_epoch(joint_module, train_loader, optimizer, device, scaler)
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
        
      
        checkpoint_path = os.path.join(checkpoint_folder, f"best_checkpoint_epoch_{epoch+1}.pth")
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1} with valid loss {valid_loss:.4f}")
