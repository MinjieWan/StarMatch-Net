# StarMatch-Net
## Abstract

This is the official implementation of **StarMatch-Net**, a star map matching method based on dual-branch feature fusion.

---
## Project Structure
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
---
## Requirements
The project is tested with the following environment:
- Python == 3.10.19
- Numpy ==2.1.2
- PyTorch ==2.6.0
---
## Data Preparation
We project star points onto the image plane using a camera model and extract their centroid coordinates, which are saved in `.txt` files. Our dataset is stored in https://drive.google.com/drive/folders/14TJsUhy1b5qH-AooPPhurXGsEfzIYwdl?usp=drive_link
```text
# data example
train/processed_truth_000001.txt
train/processed_truth_000002.txt
……
```
---
## Checkpoints

Our trained weight file `checkpoint.pt` is stored in https://drive.google.com/drive/folders/14TJsUhy1b5qH-AooPPhurXGsEfzIYwdl?usp=drive_link

---
## Usage
1.Train
```text
python train.py
```
2.Evaluation and Inference
```text
python test.py
```
---
## LICENSE
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

