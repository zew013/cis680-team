import glob
import os
import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image

from pycocotools import mask as mask_utils

import json
import numpy as np
from tqdm import tqdm

# new_height, new_width  = 160, 256
new_height, new_width  = 256, 256
input_transforms = transforms.Compose([
    transforms.Resize((new_height, new_width), antialias=True),
    transforms.ToTensor(),
])

target_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((new_height, new_width), antialias=True),
])

def resize_bbox(bbox, original_width, original_height, new_width=256, new_height=160):
    """
    Resize bounding box coordinates based on the new image dimensions.

    Args:
        bbox (list or tuple): The original bounding box [x, y, width, height].
        original_width (int or float): The original width of the image.
        original_height (int or float): The original height of the image.
        new_width (int or float): The new width to resize the image to.
        new_height (int or float): The new height to resize the image to.

    Returns:
        list: The resized bounding box [new_x, new_y, new_width, new_height].
    """
    # Calculate scaling factors for width and height
    scale_x = new_width / original_width
    scale_y = new_height / original_height

    # Unpack the original bounding box
    x, y, w, h = bbox

    # Apply scaling factors to each component of the bounding box
    new_x = x * scale_x
    new_y = y * scale_y
    new_w = w * scale_x
    new_h = h * scale_y

    return [new_x, new_y, new_x + new_w, new_y + new_h]



class SA1B_Dataset(torchvision.datasets.ImageFolder):
    """A data loader for the SA-1B Dataset from "Segment Anything" (SAM)
    This class inherits from :class:`~torchvision.datasets.ImageFolder` so
    the same methods can be overridden to customize the dataset.
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.imgs[index] # discard automatic subfolder labels
        image = self.loader(path)
        masks = json.load(open(f'{path[:-3]}json'))['annotations'] # load json masks
        # print(json.load(open(f'{path[:-3]}json')))
        seg = []
        bbox = []

        for m in masks:
            # decode masks from COCO RLE format
            seg_piece = mask_utils.decode(m['segmentation'])
            original_height, original_width = seg_piece.shape
            seg.append(seg_piece)
            bbox.append(resize_bbox(m['bbox'], original_width, original_height, new_width=new_width, new_height=new_height))
            
        seg = np.stack(seg, axis=-1)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            seg = self.target_transform(seg)
        seg[seg > 0] = 1 # convert to binary masks

        return image, torch.tensor(bbox), torch.tensor(seg).long()
        

    def __len__(self):
        return len(self.imgs)


input_reverse_transforms = transforms.Compose([
    transforms.ToPILImage(),
])

import matplotlib.pyplot as plt
def show_image(image, bbox, masks, num_rows=12, num_cols=12):
    # image: numpy image
    # target: mask [N, H, W]
    # fig, axs = plt.subplots(row, col, figsize=(20, 12))
    # for i in range(row):
    #     for j in range(col):
    #         if i*row+j < target.shape[0]:
    #             axs[i, j].imshow(image)
    #             axs[i, j].imshow(target[i*row+j], alpha=0.5)
    #         else:
    #             axs[i, j].imshow(image)
    #         axs[i, j].axis('off')

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 12))
    total = num_rows * num_cols
    
    for i in range(num_rows):
        for j in range(num_cols):
            idx = i * num_cols + j
            ax = axs[i, j]
            if idx < masks.shape[0]:
                ax.imshow(image)
                ax.imshow(masks[idx], alpha=0.5, cmap='jet')  # Use a colormap for better visibility
                # Draw bounding box
                box = bbox[idx].numpy()
                rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                     linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            else:
                ax.imshow(image)
            ax.axis('off')
    plt.tight_layout()
    plt.show()


from finetune_lora import *
import random
from torch.utils.data import Subset

path = './sa1b'
dataset = SA1B_Dataset(path, transform=input_transforms, target_transform=target_transforms)
# image, bbox, masks = dataset.__getitem__(3535)

# Set a manual seed for reproducibility
torch.manual_seed(42)

subset_ratio = 0.3
subset_size = int(subset_ratio * len(dataset))
torch.manual_seed(42)
subset_indices = random.sample(range(len(dataset)), subset_size)

subset_dataset = Subset(dataset, subset_indices)

from torch.utils.data import random_split
train_ratio = 0.8
val_ratio = 0.2

# Calculate the number of samples for each set
train_size = int(train_ratio * len(subset_dataset))
val_size = len(subset_dataset) - train_size

print(f"Total samples: {len(subset_dataset)}")
print(f"Training samples: {train_size}")
print(f"Validation samples: {val_size}")


# Split the dataset
train_dataset, val_dataset = random_split(subset_dataset, [train_size, val_size])

print(f"Training dataset length: {len(train_dataset)}")
print(f"Validation dataset length: {len(val_dataset)}")

NUM_GPUS = torch.cuda.device_count()
DEVICE = 'cuda'
print(NUM_GPUS, DEVICE)
from types import SimpleNamespace
from pytorch_lightning.strategies import DDPStrategy

def dict_to_namespace(d):
    return SimpleNamespace(**d)

config_dict = {
    "model_type": "vit_b",
    "checkpoint_path": "sam_vit_b_01ec64.pth",
    "freeze_image_encoder": False,
    "freeze_prompt_encoder": False,
    "freeze_mask_decoder": False,
    "batch_size": 4,
    # 'original_img_size': 1024,
    'patch_size': 16,
    # 'new_img_size': (256, 256),
    "steps": 224,
    "learning_rate": 1e-4,
    "weight_decay": 1e-2,
    "metrics_interval": 300,
    "output_dir": 'project_checkpoints-v3/'
}

args = dict_to_namespace(config_dict)

model = SAMFinetuner(
    args.model_type,
    args.checkpoint_path,
    freeze_image_encoder=args.freeze_image_encoder,
    freeze_prompt_encoder=args.freeze_prompt_encoder,
    freeze_mask_decoder=args.freeze_mask_decoder,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    metrics_interval=args.metrics_interval,
    # original_img_size = 1024,
    patch_size = 16,
    # new_img_size = (256, 256)
)

callbacks = [
    LearningRateMonitor(logging_interval='epoch'),
    ModelCheckpoint(
        dirpath=args.output_dir,
        filename='{step}-{val_per_mask_iou:.2f}',
        save_last=True,
        save_top_k=10,
        monitor="val_per_mask_iou",
        mode="max",
        save_weights_only=True,
        every_n_train_steps=args.metrics_interval,
    ),
]
from pytorch_lightning.loggers import TensorBoardLogger
logger = TensorBoardLogger(save_dir='lightning_logs/', name='sam_finetuning')

torch.set_float32_matmul_precision('high')

trainer = pl.Trainer(
    strategy='ddp_find_unused_parameters_true' if NUM_GPUS > 1 else 'auto',
    accelerator=DEVICE,
    devices=NUM_GPUS,
    precision=32,
    callbacks=callbacks,
    max_epochs=-1,
    max_steps=args.steps,
    val_check_interval=args.metrics_interval,
    check_val_every_n_epoch=None,
    num_sanity_val_steps=2,
    logger=logger
)

trainer.fit(model)

