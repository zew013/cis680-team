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
