# StarMatch-Net
# Abstract
In lost-in-space scenarios, the accuracy of star pattern matching methods is crucial for downstream aerospace tasks such as attitude determination. However, traditional star pattern matching methods often require preselecting guide stars and building large databases; thus an incorrect choice of guide stars may reduce matching accuracy and increase matching time. In recent years, deep learning-based star matching approaches have achieved better performance in both accuracy and efficiency, yet their robustness to noise remains insufficient. To address this issue, we propose a star pattern matching method that classifies star points using fused features within a dual-branch feature fusion network, namely StarMatch-Net. First of all, the first branch takes annular features of star points as input and employs a Transformer to capture global information of the star image. The second branch uses the angular distances between each star point and its nearest neighbors as input, and leverages a graph neural network (GNN) to aggregate neighborhood information and extract local features. Next, we use the star-wise aggregated features produced by the GNN branch to complement the Transformer’s global representation, thereby obtaining a global feature enriched with local context. Finally, we exploit a multilayer perceptron (MLP) to combine the local and global features from both branches, to perform dimensionality reduction for generating fused features, and to conduct star-point classification. The resulting per-star-point identification enables star-pattern matching for each individual star image. Experimental results show that under magnitude noise with a standard deviation of 1 magnitude, our StarMatch-Net improves the overall correct matching rate of all star points by 8% compared with the best competing approach, while reducing the accuracy variation across different test sets at the same noise level by approximately 24%.
# Project Structure
The project directory is organized as follows :
```text
StarMatch-Net/
├── Dataset/                 # Dataset folder
│   └── test/
│   └── train/
│   └── val/
├── model/                   # Model definition / modules
│   └── model.py
├── checkpoint.pt            # Saved model weights (PyTorch)
├── config.py                # Configuration file
├── train.py                 # Training script
└── test.py                  # Evaluation / inference script
```
# Requirements
The project is tested with the following environment:
- Python == 3.10.19
- Numpy ==2.1.2
- PyTorch ==2.6.0
# Data Preparation
We project star points onto the image plane using a camera model and extract their centroid coordinates, which are saved in `.txt` files. Our dataset is stored in https://drive.google.com/drive/folders/14TJsUhy1b5qH-AooPPhurXGsEfzIYwdl?usp=drive_link
```text
# data example
train/processed_truth_000001.txt
train/processed_truth_000002.txt
……
```
# Checkpoints
Our trained weight file `checkpoint.pt` is stored in https://drive.google.com/drive/folders/14TJsUhy1b5qH-AooPPhurXGsEfzIYwdl?usp=drive_link
# Usage
1.Train
```text
python train.py
```
2.Evaluation and Inference
```text
python test.py
```

