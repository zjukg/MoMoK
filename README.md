# Mixture of Modality Knowledge Experts for Robust Multi-modal Knowledge Graph Completion


## 🌈 Overview
![model](resource/model.png)

## 🔬 Dependencies
- Python==3.9
- numpy==1.24.2
- scikit_learn==1.2.2
- torch==2.0.0
- tqdm==4.64.1

## 💻 Data preparation
The multi-model embedding of MMKGs are too large so you should download them from the [Google Drive Link](https://drive.google.com/file/d/1nRHdeWiVi9d_FKli3x7sO87ARasad39w/view?usp=sharing). Please unzip the embedding files and put them in the corresponding path in `datasets/`



## 📕 Train and Evaluation

You can refer to the training scripts in `scripts/train.sh` to reproduce our experiment results. Here is an example for DB15K dataset.

```bash
nohup python train.py --cuda 0 --lr 0.001 --mu 0.0001 --dim 200 --dataset MKG-W --epochs 2000 > log.txt &

nohup python train.py --cuda 1 --lr 0.0005 --mu 0.0001 --dim 300 --dataset MKG-Y --epochs 2000 > log.txt &
```
The evaluation results will be printed in the command line after training.

🤝 Cite:
```
@misc{zhang2024mixture,
      title={Mixture of Modality Knowledge Experts for Robust Multi-modal Knowledge Graph Completion}, 
      author={Yichi Zhang and Zhuo Chen and Lingbing Guo and Yajing Xu and Binbin Hu and Ziqi Liu and Wen Zhang and Huajun Chen},
      year={2024},
      eprint={2405.16869},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```

