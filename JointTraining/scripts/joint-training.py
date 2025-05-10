# !pip install transformers
# !pip install torch
# !pip install torchvision
# !pip install lightning
# !pip install sklearn
# !pip install ipywidgets
# !pip install wandb

# srun --jobid=57269736 --pty bash -i
# pip install transformers torch torchvision lightning sklearn ipywidgets wandb==0.18.0

import torch
import torch.nn as nn
from transformers import Dinov2Model
import json
import torch
from safetensors.torch import load_file
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import os
import wandb
from dataclasses import dataclass
# Sets the internal precision of float32 matrix multiplications.
torch.set_float32_matmul_precision('high')
 
# To enable determinism.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
 
# # To render the matplotlib figure in the notebook.
# %matplotlib inline


os.environ["WANDB_API_KEY"] = "4f8dccbaced16f201316dd4113139739694dfd3b"


# Initialize wandb and log the configuration parameters.
wandb.init(
    project="joint-training",
    name="joint-training-exp-1", #args.name,  
    config={}
    )



class SimCLR(nn.Module):
    def __init__(self, dropout_p=0.5, embedding_size=128, freeze=False, linear_eval=False):
        super().__init__()
        self.linear_eval = linear_eval
        self.dropout_p = dropout_p
        self.embedding_size = embedding_size

        # Load the DINOv2 model
        self.encoder = Dinov2Model.from_pretrained('microsoft/rad-dino')
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(768, 256),  # DINOv2 base model has an embedding dimension of 768
            nn.Dropout(p=self.dropout_p),
            nn.ReLU(),
            nn.Linear(256, embedding_size)
        )

    def forward(self, x):
        if not self.linear_eval:
            # We expect x to be a list of two views
            # We concatenate both views to be one large batch
            # of size 2*batch_size, i.e., (2*B, C, W, H)
            x = torch.cat(x, dim=0)

        # DINOv2 expects inputs of shape (batch_size, num_channels, height, width)
        # Ensure x is properly normalized and resized as required by DINOv2

        outputs = self.encoder(x)
        encoding = outputs.last_hidden_state[:, 0]  # Extract the [CLS] token representation

        # If not linear_eval: Projections: (2*B, E), they are concatenated
        # Else: (B, E)
        projection = self.projection(encoding)

        return projection


retrieval_checkpoint = '/home/saahmed/scratch/projects/Image-segmentation/retrieval/checkpoints/simclr_rad-dino_pos-pairs_aug-pairs_epochs100/best_model/'

retrieval_model = SimCLR(dropout_p=0.3, embedding_size=128, freeze=False, linear_eval=True)

encoder_state_dict = load_file(retrieval_checkpoint+'model.safetensors')
retrieval_model.encoder.load_state_dict(encoder_state_dict)

projection_state_dict = torch.load(retrieval_checkpoint+'projection_head.pth', map_location=torch.device('cpu'))
retrieval_model.projection.load_state_dict(projection_state_dict)


retrieval_model.to('cuda')


import torch
import torch.nn as nn
import torch.nn.functional as F
 
# Importing lighting along with a built-in callback it provides.
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint,TQDMProgressBar
 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from IPython.display import display, clear_output

# We assume that your segmentation model is Segformer.
# Since the composite input will have more channels (e.g., query (3) + 2 guide images (3 each) + 2 masks (1 each) = 11 channels),
# we wrap the segformer with an input adapter to map the 11 channels to 3 channels.
class JointSegmentationModel(nn.Module):
    def __init__(self, base_model, composite_in_channels=11):
        super().__init__()
        self.input_adapter = nn.Conv2d(composite_in_channels, 3, kernel_size=1)  # maps 11 channels to 3 channels
        self.base_model = base_model  # e.g., a SegformerForSemanticSegmentation instance

    def forward(self, x):
        # x: (B, composite_in_channels, H, W)
        x = self.input_adapter(x)
        outputs = self.base_model(pixel_values=x, return_dict=True)
        # Upsample logits to match the input resolution
        logits = F.interpolate(outputs["logits"], size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits

# ----------------------------
# Loss Function
# ----------------------------

def dice_coef_loss(predictions, ground_truths, num_classes=4, dims=(1, 2), smooth=1e-8):
    """
    Computes a combined Dice coefficient and Cross-Entropy loss.
    predictions: (B, num_classes, H, W)
    ground_truths: (B, H, W) with integer labels in [0, num_classes-1]
    """
    ground_truth_oh = F.one_hot(ground_truths, num_classes=num_classes)  # (B, H, W, num_classes)
    prediction_norm = F.softmax(predictions, dim=1).permute(0, 2, 3, 1)    # (B, H, W, num_classes)
    intersection = (prediction_norm * ground_truth_oh).sum(dim=dims)
    summation = prediction_norm.sum(dim=dims) + ground_truth_oh.sum(dim=dims)
    dice = (2.0 * intersection + smooth) / (summation + smooth)
    dice_mean = dice.mean()
    CE = F.cross_entropy(predictions, ground_truths)
    return (1.0 - dice_mean) + CE

# ----------------------------
# Differentiable Top-2 Retrieval
# ----------------------------

def differentiable_top2(similarity, tau=1.0):
    """
    Given similarity scores of shape (B, N), returns two one-hot weight vectors (B, N)
    using Gumbel-softmax to approximate the top-2 selections in a differentiable manner.
    """
    weights1 = F.gumbel_softmax(similarity, tau=tau, hard=True)  # first selection (one-hot)
    B, N = similarity.shape
    masked_similarity = similarity.clone()
    # For each batch element, mask the index selected in weights1
    indices = weights1.argmax(dim=-1, keepdim=True)  # shape (B, 1)
    for i in range(B):
        masked_similarity[i, indices[i]] = float('-inf')
    weights2 = F.gumbel_softmax(masked_similarity, tau=tau, hard=True)  # second selection
    return weights1, weights2

# ----------------------------
# Joint Training Lightning Module
# ----------------------------

class JointTrainingModule(pl.LightningModule):
    def __init__(self, retrieval_model, segmentation_model ,num_classes=4, lr=1e-4):
        """
        retrieval_model: instance of SimCLR.
        segmentation_model: instance of JointSegmentationModel.
        num_classes: number of segmentation classes.
        lr: learning rate.
        """
        super().__init__()
        self.retrieval_model = retrieval_model
        self.segmentation_model = segmentation_model
        self.lr = lr
        self.num_classes = num_classes
        
        # Initializing the required metric objects.
        self.mean_train_loss = MeanMetric()
        self.mean_valid_loss = MeanMetric()
       
 

    def forward(self, query_image, gallery_images, gallery_masks):
        """
        Retrieves two guide images (and masks) for the query image.
        query_image: (B, 3, H, W)
        gallery_images: (B, N, 3, H, W) for each query.
        gallery_masks: (B, N, H, W) for each query.
        """

        # Compute query embedding
        query_embedding = self.retrieval_model(query_image)  # (B, emb_dim)

        # Process gallery images: flatten gallery dimension to combine batch and candidate indices
        B, N, C, H, W = gallery_images.shape
        gallery_images_flat = gallery_images.view(B * N, C, H, W)


        tempbatch_size = 2
        gallery_embeddings_list = []
    
        def compute_embedding(x):
            return self.retrieval_model(x)
    
        for i in range(0, B * N, tempbatch_size):
            batch = gallery_images_flat[i:i + tempbatch_size]
            emb = torch_checkpoint(compute_embedding, batch, use_reentrant=False)
            gallery_embeddings_list.append(emb)
        gallery_embeddings_flat = torch.cat(gallery_embeddings_list, dim=0)

        # gallery_embeddings_flat = self.retrieval_model(gallery_images_flat)  # (B*N, emb_dim)
        emb_dim = gallery_embeddings_flat.shape[1]
        gallery_embeddings = gallery_embeddings_flat.view(B, N, emb_dim)  # (B, N, emb_dim)


        similarity = F.cosine_similarity(query_embedding.unsqueeze(1),  gallery_embeddings, dim=-1)  # (B, N)
        # Obtain two guide selections using differentiable top-2
        weights1, weights2 = differentiable_top2(similarity, tau=1.0)  # each of shape (B, N)

        # Retrieve guide images and masks via weighted sum (will be one-hot selections due to hard=True)
        guide_image1 = torch.einsum('bn,bnchw->bchw', weights1, gallery_images)
        guide_image2 = torch.einsum('bn,bnchw->bchw', weights2, gallery_images)
        guide_mask1 = torch.einsum('bn,bnhw->bhw', weights1, gallery_masks.float())
        guide_mask2 = torch.einsum('bn,bnhw->bhw', weights2, gallery_masks.float())

        return guide_image1, guide_image2, guide_mask1, guide_mask2

    def training_step(self, batch, batch_idx):
        """
        Expects batch as a tuple:
         - query_image: (B, 3, H, W)
         - query_mask: (B, H, W)
         - gallery_images: (B, N, 3, H, W)
         - gallery_masks: (B, N, H, W)
        """
        query_image, query_mask, gallery_images, gallery_masks = batch

        # Retrieve the two guide images and masks
        guide_image1, guide_image2, guide_mask1, guide_mask2 = self.forward(query_image, gallery_images, gallery_masks)

        # Form composite input: concatenate along channel dimension
        # Query image (3), guide image1 (3), guide image2 (3), guide mask1 (1), guide mask2 (1) = 11 channels total.
        composite_input = torch.cat([
            query_image,
            guide_image1,
            guide_image2,
            guide_mask1.unsqueeze(1),
            guide_mask2.unsqueeze(1)
        ], dim=1)  # (B, 11, H, W)

        # Forward pass through segmentation model
        pred_masks = self.segmentation_model(composite_input)  # (B, num_classes, H, W)
        # Compute loss using Dice + Cross-Entropy loss
        loss = dice_coef_loss(pred_masks, query_mask, num_classes=self.num_classes)

        self.mean_train_loss.update(loss, weight=composite_input.shape[0])

        self.log("train/batch_loss", self.mean_train_loss, prog_bar=True, logger=False)
        
        return loss

    def on_train_epoch_end(self):
        # Compute the mean training loss 
        train_loss = self.mean_train_loss.compute()
        
        # Log the metrics for display in the progress bar and logger
        self.log("train/loss", train_loss,)
        self.log("epoch", self.current_epoch)

        # Print the metrics to the console
        print(f"Epoch {self.current_epoch}: Train Loss: {train_loss}") 

        # Log metrics to wandb
        wandb.log({
            "epoch": self.current_epoch,
            "Train Loss": train_loss,
        })
        
        # Reset the metrics for the next epoch
        self.mean_train_loss.reset()

    def validation_step(self, batch, batch_idx):
        query_image, query_mask, gallery_images, gallery_masks = batch
        guide_image1, guide_image2, guide_mask1, guide_mask2 = self.forward(query_image, gallery_images, gallery_masks)
        composite_input = torch.cat([
            query_image,
            guide_image1,
            guide_image2,
            guide_mask1.unsqueeze(1),
            guide_mask2.unsqueeze(1)
        ], dim=1)
        pred_masks = self.segmentation_model(composite_input)
        loss = dice_coef_loss(pred_masks, query_mask, num_classes=self.num_classes)

        
        self.mean_valid_loss.update(loss, weight=composite_input.shape[0])
        
    def on_validation_epoch_end(self):
        # Compute the mean validation loss
        valid_loss = self.mean_valid_loss.compute()
        
        
        # Log the metrics for display in the progress bar and logger
        self.log("valid/loss", valid_loss, prog_bar=True)
        self.log("epoch", self.current_epoch)

        # Print the metrics to the console
        print(f"Epoch {self.current_epoch}: Valid Loss: {valid_loss}") 
        # Log metrics to wandb
        wandb.log({
            "epoch": self.current_epoch,
            "Train Loss": valid_loss,
        })
        # Reset the metrics for the next epoch
        self.mean_valid_loss.reset()
     

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.retrieval_model.parameters()) + list(self.segmentation_model.parameters()),
            lr=self.lr
        )
        return optimizer


import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from tqdm import tqdm
# Importing torchmetrics modular and functional implementations.
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassF1Score


def convert_to_rgb(img):
    return img.convert("RGB")
    
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Lambda(convert_to_rgb),
    transforms.ToTensor(),
])


import torch

def remap_labels(target, mapping = {0: 0, 85: 1, 170: 2, 255: 3}):
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



image_dir_ed = '/home/saahmed/scratch/projects/Image-segmentation/datasets/ACDC/processed_data/Training/ed/images/'
mask_dir_ed = '/home/saahmed/scratch/projects/Image-segmentation/datasets/ACDC/processed_data/Training/ed/masks/'


image_dir_es = '/home/saahmed/scratch/projects/Image-segmentation/datasets/ACDC/processed_data/Training/es/images/'
mask_dir_es = '/home/saahmed/scratch/projects/Image-segmentation/datasets/ACDC/processed_data/Training/es/masks/'


image_filenames = sorted([os.path.join(image_dir_ed, i) for i in os.listdir(image_dir_ed)]) + \
                           sorted([os.path.join(image_dir_es, i) for i in os.listdir(image_dir_es)])
mask_filenames = sorted([os.path.join(mask_dir_ed, i) for i in os.listdir(mask_dir_ed)]) + \
                          sorted([os.path.join(mask_dir_es, i) for i in os.listdir(mask_dir_es)])



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

for img_path, mask_path in tqdm(zip(image_filenames,mask_filenames)):
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



database_images = []
database_masks = []

for img , mask in tqdm(zip(valid_images,valid_masks)):
    g_img = Image.open(img)
    g_mask = Image.open(mask).convert('L')

    processed_img = preprocess(g_img)
    # with torch.no_grad():
    #     processed_img_emb = retrieval_model(processed_img.unsqueeze(0).to('cuda'))
    #     processed_img_emb = processed_img_emb.cpu()  # Move back to CPU to free up GPU memory

    g_mask = TF.resize(g_mask, (256,256), interpolation=Image.NEAREST)
    
    g_mask = torch.from_numpy(np.array(g_mask)).long()
    g_mask = remap_labels(g_mask)

    database_images.append(processed_img)
    database_masks.append(g_mask)
    # database_images_embeddings.append(processed_img_emb)

# gallery_images: (num_gallery, 3, H, W) and gallery_masks: (num_gallery, H, W)
database_images = torch.stack(database_images, dim=0)#.unsqueeze(0)
database_masks = torch.stack(database_masks, dim=0)#.unsqueeze(0)
# database_images_embeddings = torch.stack(database_images_embeddings, dim=0)



import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class JointMedicalDataset(Dataset):
    def __init__(self, image_file_names, mask_file_names, database_images, database_masks, image_size=(256, 256)):

        self.database_images = database_images
        self.database_masks = database_masks

       
        self.image_filenames = image_file_names
        self.mask_filenames  = mask_file_names
        assert len(self.image_filenames) == len(self.mask_filenames), "Number of images and masks do not match"
        
        self.image_size = image_size

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load query image and mask
        query_image_path = self.image_filenames[idx]
        query_mask_path = self.mask_filenames[idx]

        query_image = Image.open(query_image_path)
        query_mask = Image.open(query_mask_path).convert('L')
        
        # query_image = TF.resize(query_image, self.image_size)
        query_mask = TF.resize(query_mask, self.image_size, interpolation=Image.NEAREST)
        
        # query_image = TF.to_tensor(query_image)
        # query_image = TF.normalize(query_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        query_image = preprocess(query_image)
        query_mask = torch.from_numpy(np.array(query_mask)).long()
        query_mask = remap_labels(query_mask)

     
        return query_image, query_mask, database_images, database_masks



# train_dataset = JointMedicalDataset(train_imgs, train_masks, database_images, database_masks)
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)


# val_dataset = JointMedicalDataset(val_imgs, val_masks, database_images, database_masks)
# val_loader = DataLoader(val_dataset, batch_size=8)

@dataclass(frozen=True)
class Paths:
    DATA_TRAIN_IMAGES = train_imgs
    DATA_TRAIN_LABELS = train_masks

    DATA_VALID_IMAGES = val_imgs
    DATA_VALID_LABELS = val_masks

    Guide_database_imgs = database_images
    Guide_database_masks = database_masks



class MedicalSegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        num_classes=4,
        img_size=(256, 256),
        ds_mean=(0.485, 0.456, 0.406),
        ds_std=(0.229, 0.224, 0.225),
        batch_size=8,
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


        train_imgs = Paths.DATA_TRAIN_IMAGES
        train_msks = Paths.DATA_TRAIN_LABELS
        
        valid_imgs =  Paths.DATA_VALID_IMAGES
        valid_msks = Paths.DATA_VALID_LABELS

        # test_imgs =  Paths.DATA_TEST_IMAGES
        # test_msks = Paths.DATA_TEST_LABELS
        guide_images = Paths.Guide_database_imgs
        guide_masks = Paths.Guide_database_masks
 
        self.train_ds =  JointMedicalDataset(train_imgs, train_msks, guide_images, guide_masks)
 
        self.valid_ds =  JointMedicalDataset(valid_imgs, valid_msks, guide_images, guide_masks)
 

        # self.test_ds = MedicalDataset(image_paths=test_imgs, mask_paths=test_msks, img_size=self.img_size, 
        #                                is_train=False, ds_mean=self.ds_mean, ds_std=self.ds_std)
 
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


    # def test_dataloader(self):
    #     # Create validation dataloader object.
    #     return DataLoader(
    #         self.test_ds, batch_size=self.batch_size,  pin_memory=self.pin_memory, 
    #         num_workers=self.num_workers, shuffle=self.shuffle_validation
    #     )

# # Create the segmentation model using a pretrained segformer.
# segformer_base = SegformerForSemanticSegmentation.from_pretrained(
#     "nvidia/segformer-b0-finetuned-ade-512-512",
#     num_labels=4,
#     ignore_mismatched_sizes=True
# )


# %%time
 
dm = MedicalSegmentationDataModule(
    num_classes=4,
    img_size=(256, 256),
    batch_size=8,
    num_workers=0,
    shuffle_validation=False,
)
 

# Create training & validation dataset.
dm.setup()
 
train_loader, valid_loader = dm.train_dataloader(), dm.val_dataloader()


ckpt_path = '/scratch/saahmed/projects/Image-segmentation/segmentation/lightning_logs/version_0/checkpoints/ckpt_053-vloss_0.1769_vf1_0.9259.ckpt'
checkpoint = torch.load(ckpt_path, map_location="cuda:0")
state_dict = checkpoint["state_dict"]

# Remove any unwanted prefixes from the state_dict keys
new_state_dict = {key.replace("model.", ""): value for key, value in state_dict.items()}

# Load the configuration from a pretrained model and update the number of classes
config = SegformerForSemanticSegmentation.config_class.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
config.num_labels = 4  # Update to the number of classes used in your checkpoint

# Instantiate the model with the updated configuration
segformer_base = SegformerForSemanticSegmentation(config)

# Load the state dictionary into the model
segformer_base.load_state_dict(new_state_dict)


segmentation_model = JointSegmentationModel(segformer_base, composite_in_channels=11)


joint_module = JointTrainingModule(retrieval_model, segmentation_model ,num_classes=4, lr=1e-4)


from lightning.pytorch.callbacks import TQDMProgressBar


 # Seed everything for reproducibility.
pl.seed_everything(42, workers=True)
 

trainer = pl.Trainer(
    max_epochs=50,
    accelerator="auto",  # Auto select the best hardware accelerator available
    devices='auto',  # Auto select available devices for the accelerator (For eg. mutiple GPUs)
    strategy="auto",  # Auto select the distributed training strategy.
    precision="16-mixed",  # Using Mixed Precision training.
    callbacks=[TQDMProgressBar(refresh_rate=10)],
    enable_progress_bar=True,
    log_every_n_steps=1)

trainer.fit(joint_module, train_loader, valid_loader)
