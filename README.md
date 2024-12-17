# **Enhancing SAM Performance on Low-Resolution Images**

## **Overview**

This project enhances the **Segment Anything Model (SAM)** for low-resolution images by integrating:

1. **Low-Rank Adaptation (LoRA)** for efficient fine-tuning, reducing computational costs while maintaining segmentation performance.
2. **Normalized Cuts (NCut)** for feature visualization, diagnosing performance issues across different resolutions.

These improvements help SAM adapt to real-world applications where low-resolution inputs are common, such as edge devices and bandwidth-limited scenarios.

---

## **Key Features**

- **LoRA Fine-Tuning**:
  - Reduces trainable parameters from **91M** to **28M**, lowering computational overhead.
  - Enables efficient adaptation of SAM for low-resolution images.

- **NCut Diagnostics**:
  - Visualizes model features to understand discrepancies caused by low-resolution inputs.
  - Provides insights into SAM's behavior across different layers and resolutions.

- **Low-Resolution Adaptations**:
  - **Dynamic Positional Encoding Interpolation**: Adjusts positional encodings for different input resolutions.
  - **Custom Preprocessing**: Normalizes images and resizes them to target resolutions (e.g., 256x256).
  - **Post-Processing**: Refines segmentation masks by removing artifacts and duplicate regions.

---
Full_Segment_Anything: modified SAM directory that could take customerized image input sizes. Modification done on trained embeddings through interpolation (down-sampling).

ncut_pytorch: modified ncut forked from https://github.com/huzeyann/ncut_pytorch that could take intermediate results from our modified models.

segment_anything: Meta implmentation of segment anything https://github.com/facebookresearch/segment-anything

### **finetune_v2.py and finetune_lora.py**

This script fine-tunes the **Segment Anything Model (SAM)** to improve segmentation on low-resolution images. It employs **PyTorch Lightning** for streamlined training, logging, and checkpointing. The script allows freezing specific model components (image encoder, prompt encoder, mask decoder) to optimize efficiency. The training process uses a combination of **Focal Loss**, **Dice Loss**, and **IoU Loss** to enhance mask prediction accuracy. During training, the script tracks IoU scores and losses, saving the best-performing model based on validation metrics.

### **train_v2.py and train_lora.py**

This script resizes images and bounding boxes to a resolution of **256x256** and converts segmentation masks to binary masks. The dataset is split into **80% training** and **20% validation** subsets. Model checkpoints and metrics are saved periodically, and training supports multi-GPU setups using the Distributed Data Parallel (DDP) strategy.

### **ncut.ipynb**
All experiments using ncut to visualize different versions of SAM models. Could run all experiment in one click.

