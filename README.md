# UNet Lung Segmentation

This project implements a UNet-based deep learning pipeline for automatic lung segmentation using chest X-ray images from public datasets. The pipeline includes preprocessing, model training, and bulk segmentation using a trained model.

## 📁 Project Structure
```
├── requirements.ipynb
├── preprocessing.ipynb
├── train.ipynb
├── bulk_segmentation.ipynb
├── score.ipynb (optional)
├── data/ # Contains the preprocessed .npy files
├── models/ # Contains saved trained models (.keras)
├── temp/ # Contains the best trained model from train.ipynb
├── input/ # Place test lung images here for segmentation
├── segmented/ # Segmentation results will be saved here
```
---

## 📌 Notes
- For best results, use the aggregatedSet.
- You can use our pre-trained models directly for inference without retraining i.e. skip to Step 4. You can download all the pretrained models in this link: https://drive.google.com/drive/folders/18HLHHZ3bmFwD5JALmLQNE6TaE7I4c38E?usp=sharing

## 🚀 How to Use

### 1. Install Dependencies
Run the notebook:

`requirements.ipynb`

This will install all required packages via `pip`. Make sure `pip` or `pip3` is installed on your system.

---

### 2. Preprocess the Data
Run:

`preprocessing.ipynb`

Here, you can:
- **Set the image resolution**, default is `(256, 256)`.
- **Choose the dataset**:
  - `montgomerySet` (~140 image-mask pairs)
  - `shenzhenSet` (~600 image-mask pairs)
  - `darwinSet` (~6000 image-mask pairs)
  - **Default**: `aggregatedSet` (combination of all three)

This step outputs four `.npy` files inside the `data/` folder:
- `Unet-Train-Lung-Images.npy`, `Unet-Validate-Lung-Images.npy`
- `Unet-Train-Lung-Masks.npy`, `Unet-Validate-Lung-Masks.npy`

---

### 3. Train the Model
Run:

`train.ipynb`

Options:
- You can choose to train on wavelet-transformed masks **(not recommended)** as it yielded inferior results in our experiments.
- By default, it uses the preprocessed datasets from the previous step.

The best model is saved as a `.keras` file inside the `temp/` directory.

---

### 4. Perform Bulk Segmentation
Run:

`bulk_segmentation.ipynb`

Instructions:
- Place the chest X-ray images you want to segment into the `input/` folder.
- The notebook will load the best model from `temp/` and generate the segmented outputs in the `segmented/` folder.

---

### 5. (Optional) Evaluate Model Performance
Run:

`score.ipynb`

This notebook computes the **DICE scores** for each trained model found in the `models/` folder.
