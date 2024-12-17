import os
import argparse
import sys
from collections import defaultdict, deque
import pickle

import numpy as np
from PIL import Image
import cv2

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.distributed as dist
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import segmentation_models_pytorch as smp

from transformers.models.maskformer.modeling_maskformer import dice_loss, sigmoid_focal_loss

# Add the SAM directory to the system path
sys.path.append("./segment-anything")
from segment_anything import sam_model_registry

NUM_WORKERS = 15  # https://github.com/pytorch/pytorch/issues/42518
NUM_GPUS = torch.cuda.device_count()
DEVICE = 'cuda'


# Source: https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/comm.py
def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


# Source: https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/comm.py
def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def collate_fn(batch):
    images, bboxes, masks = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, bboxes, masks


def interpolate_pos_embed(model, new_grid_size):
    """
    Interpolates the positional embeddings to a new grid size.
    
    Args:
        model (ImageEncoderViT): The image encoder model.
        new_grid_size (tuple): The new grid size as (new_H, new_W).
    
    Returns:
        nn.Parameter: The interpolated positional embeddings.
    """
    pos_embed = model.pos_embed  # Shape: (1, H, W, C)
    _, H, W, C = pos_embed.shape  # Original grid size, e.g., H=W=64 for 1024x1024
    
    new_H, new_W = new_grid_size  # e.g., (10, 16) for 160x256 with patch size 16
    
    # Permute to (1, C, H, W) for interpolation
    pos_embed = pos_embed.permute(0, 3, 1, 2)  # Shape: (1, C, H, W)
    
    # Interpolate to new grid size
    pos_embed = F.interpolate(pos_embed, size=(new_H, new_W), mode='bicubic', align_corners=False)
    
    # Permute back to (1, H, W, C)
    pos_embed = pos_embed.permute(0, 2, 3, 1)  # Shape: (1, new_H, new_W, C)
    
    # Convert to nn.Parameter
    pos_embed = nn.Parameter(pos_embed)
    
    return pos_embed


class SAMFinetuner(pl.LightningModule):

    def __init__(
            self,
            model_type,
            checkpoint_path,
            freeze_image_encoder=False,
            freeze_prompt_encoder=False,
            freeze_mask_decoder=False,
            batch_size=4,
            learning_rate=1e-4,
            weight_decay=1e-4,
            train_dataset=None,
            val_dataset=None,
            metrics_interval=10,
            original_img_size = 1024,
            patch_size = 16,
            new_img_size = (256, 256),
        ):
        super(SAMFinetuner, self).__init__()

        self.model_type = model_type
        self.model = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
        self.model.to(device=self.device)

        new_grid_size = (new_img_size[0] // patch_size, new_img_size[1] // patch_size)  # (10, 16)

        # Interpolate pos_embed
        self.model.image_encoder.pos_embed = interpolate_pos_embed(self.model.image_encoder, new_grid_size)
        
        # Update the image encoder's img_size attribute
        self.model.image_encoder.img_size = new_img_size[0]  # Update image size to (160, 256)

        prompt_embed_dim = 256
        vit_patch_size = 16
        image_embedding_size = new_img_size[0] // vit_patch_size
        image_embedding_size_width_unusednow = new_img_size[1] // vit_patch_size
        
        from segment_anything.modeling import PromptEncoder
        self.model.prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(new_img_size[0], new_img_size[1]),
            mask_in_chans=16,
        )

        
        self.freeze_image_encoder = freeze_image_encoder
        if freeze_image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if freeze_prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if freeze_mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_metric = defaultdict(lambda: deque(maxlen=metrics_interval))
        self.val_metric = []
        self.metrics_interval = metrics_interval

    def forward(self, imgs, bboxes, labels):
        _, _, H, W = imgs.shape
        features = self.model.image_encoder(imgs)
        num_masks = sum([len(b) for b in bboxes])

        loss_focal = loss_dice = loss_iou = 0.
        predictions = []
        tp, fp, fn, tn = [], [], [], []
        for feature, bbox, label in zip(features, bboxes, labels):
            # Embed prompts
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=bbox,
                masks=None,
            )
            # Predict masks
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=feature.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            # Upscale the masks to the original image resolution
            masks = F.interpolate(
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )
            predictions.append(masks)
            # Compute the iou between the predicted masks and the ground truth masks
            batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(
                masks,
                label.unsqueeze(1),
                mode='binary',
                threshold=0.5,
            )
            batch_iou = smp.metrics.iou_score(batch_tp, batch_fp, batch_fn, batch_tn)
            # Compute the loss
            masks = masks.squeeze(1).flatten(1)
            label = label.flatten(1)
            loss_focal += sigmoid_focal_loss(masks, label.float(), num_masks)
            loss_dice += dice_loss(masks, label.float(), num_masks)
            loss_iou += F.mse_loss(iou_predictions, batch_iou, reduction='sum') / num_masks
            tp.append(batch_tp)
            fp.append(batch_fp)
            fn.append(batch_fn)
            tn.append(batch_tn)
        return {
            'loss': 20. * loss_focal + loss_dice + loss_iou,  # SAM default loss
            'loss_focal': loss_focal,
            'loss_dice': loss_dice,
            'loss_iou': loss_iou,
            'predictions': predictions,
            'tp': torch.cat(tp),
            'fp': torch.cat(fp),
            'fn': torch.cat(fn),
            'tn': torch.cat(tn),
        }
    
    def training_step(self, batch, batch_nb):
        imgs, bboxes, labels = batch
        outputs = self(imgs, bboxes, labels)

        for metric in ['tp', 'fp', 'fn', 'tn']:
            self.train_metric[metric].append(outputs[metric])

        # aggregate step metics
        step_metrics = [torch.cat(list(self.train_metric[metric])) for metric in ['tp', 'fp', 'fn', 'tn']]
        per_mask_iou = smp.metrics.iou_score(*step_metrics, reduction="micro-imagewise")
        metrics = {
            "loss": outputs["loss"],
            "loss_focal": outputs["loss_focal"],
            "loss_dice": outputs["loss_dice"],
            "loss_iou": outputs["loss_iou"],
            "train_per_mask_iou": per_mask_iou,
        }
        self.log_dict(metrics, prog_bar=True, rank_zero_only=True)
        return metrics
    
    def validation_step(self, batch, batch_nb):
        imgs, bboxes, labels = batch
        outputs = self(imgs, bboxes, labels)
        outputs.pop("predictions")
        self.val_metric.append(outputs)
        return outputs
        
    def on_validation_epoch_start(self):
        self.val_metric = []
    
    def on_validation_epoch_end(self):
        # if NUM_GPUS > 1:
        #     outputs = all_gather(outputs)
        #     # the outputs are a list of lists, so flatten it
        #     outputs = [item for sublist in outputs for item in sublist]
        # aggregate step metics
        step_metrics = [
            torch.cat(list([x[metric].to(self.device) for x in self.val_metric]))
            for metric in ['tp', 'fp', 'fn', 'tn']]
        # per mask IoU means that we first calculate IoU score for each mask
        # and then compute mean over these scores
        per_mask_iou = smp.metrics.iou_score(*step_metrics, reduction="micro-imagewise")

        metrics = {"val_per_mask_iou": per_mask_iou}
        self.log_dict(metrics)
        return metrics
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        def warmup_step_lr_builder(warmup_steps, milestones, gamma):
            def warmup_step_lr(steps):
                if steps < warmup_steps:
                    lr_scale = (steps + 1.) / float(warmup_steps)
                else:
                    lr_scale = 1.
                    for milestone in sorted(milestones):
                        if steps >= milestone * self.trainer.estimated_stepping_batches:
                            lr_scale *= gamma
                return lr_scale
            return warmup_step_lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt,
            warmup_step_lr_builder(250, [0.66667, 0.86666], 0.1)
        )
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': "step",
                'frequency': 1,
            }
        }
    
    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            shuffle=True)
        return train_loader
    
    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            shuffle=False)
        return val_loader


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_root", type=str, required=True, help="path to the data root")
#     parser.add_argument("--model_type", type=str, required=True, help="model type", choices=['vit_h', 'vit_l', 'vit_b'])
#     parser.add_argument("--checkpoint_path", type=str, required=True, help="path to the checkpoint")
#     parser.add_argument("--freeze_image_encoder", action="store_true", help="freeze image encoder")
#     parser.add_argument("--freeze_prompt_encoder", action="store_true", help="freeze prompt encoder")
#     parser.add_argument("--freeze_mask_decoder", action="store_true", help="freeze mask decoder")
#     parser.add_argument("--batch_size", type=int, default=1, help="batch size")
#     parser.add_argument("--image_size", type=int, default=1024, help="image size")
#     parser.add_argument("--steps", type=int, default=1500, help="number of steps")
#     parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
#     parser.add_argument("--weight_decay", type=float, default=1e-2, help="weight decay")
#     parser.add_argument("--metrics_interval", type=int, default=50, help="interval for logging metrics")
#     parser.add_argument("--output_dir", type=str, default=".", help="path to save the model")

#     args = parser.parse_args()

#     # load the dataset
#     train_dataset = Coco2MaskDataset(data_root=args.data_root, split="train", image_size=args.image_size)
#     val_dataset = Coco2MaskDataset(data_root=args.data_root, split="val", image_size=args.image_size)

#     # create the model
#     model = SAMFinetuner(
#         args.model_type,
#         args.checkpoint_path,
#         freeze_image_encoder=args.freeze_image_encoder,
#         freeze_prompt_encoder=args.freeze_prompt_encoder,
#         freeze_mask_decoder=args.freeze_mask_decoder,
#         train_dataset=train_dataset,
#         val_dataset=val_dataset,
#         batch_size=args.batch_size,
#         learning_rate=args.learning_rate,
#         weight_decay=args.weight_decay,
#         metrics_interval=args.metrics_interval,
#     )

#     callbacks = [
#         LearningRateMonitor(logging_interval='step'),
#         ModelCheckpoint(
#             dirpath=args.output_dir,
#             filename='{step}-{val_per_mask_iou:.2f}',
#             save_last=True,
#             save_top_k=1,
#             monitor="val_per_mask_iou",
#             mode="max",
#             save_weights_only=True,
#             every_n_train_steps=args.metrics_interval,
#         ),
#     ]
#     trainer = pl.Trainer(
#         strategy='ddp' if NUM_GPUS > 1 else None,
#         accelerator=DEVICE,
#         devices=NUM_GPUS,
#         precision=32,
#         callbacks=callbacks,
#         max_epochs=-1,
#         max_steps=args.steps,
#         val_check_interval=args.metrics_interval,
#         check_val_every_n_epoch=None,
#         num_sanity_val_steps=0,
#     )

#     trainer.fit(model)


# if __name__ == "__main__":
#     main()
