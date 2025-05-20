# 🎮 MOEP-PCGRL

**Online Diverse Procedural Content Generation via Multi-Objective Ensemble Pruning**

> This repository provides the official implementation for the paper:
> **"Online Diverse Procedural Content Generation via Multi-Objective Ensemble Pruning"**
> The project introduces an efficient framework for generating diverse game content by adaptively pruning ensemble reinforcement learning models to match user preferences while reducing computational cost.
>
## 📄 Citation

```
@article{tong2025online,
  title={Online Diverse Procedural Content Generation via Multi-Objective Ensemble Pruning},
  author={Tong, Hao and Zhang, Qingquan, and Yuan, Bo and Wang, Handing, and Liu Jialin},
  journal={under review},
  year={2025}
}
```
---

## 🔍 Overview

Online diverse content generation is one of the important research directions in the field of procedural content generation in recent years. It can not only meet users' different preferences and enhance user experience, but also provide a large number of scenarios and problems for training and testing artificial intelligence algorithms.

Although online diverse content generation methods based on ensemble models have been proposed in literature, such methods

1. **cannot effectively meet the specific preferences of different users**
2. **require  a lot of computational resources when training and deploying individual learning models**

This paper proposes an online content generation approach based on **multi-objective ensemble pruning (MOEP-PCGRL)** that can effectively generate diverse content based on a negatively correlated ensemble reinforcement learning framework.

Our method returns a **Pareto front** of pruned ensembles, enabling flexible decision-making.

---

## 🚀 Getting Started


### Dependency Note

This project is **built upon** the [NCERL-Diverse-PCG](https://github.com/PneuC/NCERL-Diverse-PCG) framework. All environmental setups, folder structures, and model dependencies are consistent with the original repository.


### Verified Environment

- Python 3.9.6
- JPype 1.3.0
- dtw 1.4.0
- scipy 1.7.2
- torch 1.8.2+cu111
- numpy 1.20.3
- gym 0.21.0
- scipy 1.7.2
- Pillow 10.0.0
- matplotlib 3.6.3
- pandas 1.3.2
- sklearn 1.0.1

### Installation

```bash
git clone https://github.com/qingquan63/MOEP_PCGRL.git
cd MOEP_PCGRL
pip install -r requirements.txt
```

### Run a Training
```
python MOEP_PCGRL.py -- ncesac --prunning_plan 1 --randseed 1
```
