## Two-Stream Transformer for Multi-Label Image Classification

### Introduction
Reproduce PyTorch implementation of the paper "Two-Stream Transformer for Multi-Label Image Classification" ACM MM 2022 [paper](https://dl.acm.org/doi/abs/10.1145/3503161.3548343)
![alt tsformer](src/tsformer.png)

### Data Preparation
1. Download dataset and organize them as follow:
```
|datasets
|---- MSCOCO
|---- NUS-WIDE
|---- VOC2007
```
2. Preprocess using following commands:
```bash
python scripts/mscoco.py
python scripts/nuswide.py
python scripts/voc2007.py
python embedding.py --data [mscoco, nuswide, voc2007]
```

### Requirements
```
torch >= 1.9.0
torchvision >= 0.10.0
```

### Training
One can use following commands to train model.
```bash
python train.py --data mscoco --batch_size 16 --optimizer AdamW --lr 0.00001 --mode part --start_depth 9
python train.py --data nuswide --batch_size 16 --optimizer AdamW --lr 0.00001 --mode part --start_depth 1
python train.py --data voc2007 --batch_size 16 --optimizer AdamW --lr 0.00001 --mode part --start_depth 4
```

### Evaluation
Pre-trained weights can be found in [google drive](https://drive.google.com/drive/folders/1XOiLTpWHYRGR8itp4aqQZsbXWHV_TT0j?usp=sharing). Download and put them in the `experiments` folder, then one can use follow commands to reproduce results reported in paper.

```bash
python evaluate.py --exp experiments/TSFormer_mscoco/exp1    # Microsoft COCO
python evaluate.py --exp experiments/TSFormer_nuswide/exp1   # NUS-WIDE
python evaluate.py --exp experiments/TSFormer_voc2007/exp1   # Pascal VOC 2007
```

### Main Results
|  dataaset   | mAP  | ours |
|  ---------  | ---- | ---- | 
| VOC 2007    | 97.0 | 97.0 |
| MS-COCO     | 88.9 | 88.9 |
| NUS-WIDE    | 69.3 | 69.3 |

# BibTex
```
@inproceedings{zhu2022two,
  title={Two-Stream Transformer for Multi-Label Image Classification},
  author={Zhu, Xuelin and Cao, Jiuxin and Ge, Jiawei and Liu, Weijia and Liu, Bo},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={3598--3607},
  year={2022}
}