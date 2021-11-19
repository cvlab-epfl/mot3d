# Fast Single and Multi Camera Multi-object Tracking using Minimum Cost Maximum Flow Algorithm

This package provides all the necessary routines to create and solve minimum cost maximum flow graphs for 2D or 3D multi object tracking.
In the projects folder we provide solutions for single view and multiple view online multi-object tracking for the PETS2009 sequence.

#### Solvers:
- Integer Linear Programming (ILP) [Slow]
- Minimum-update Successive Shortest Path (muSSP) [Very Fast] https://github.com/yu-lab-vt/muSSP

Big thanks to authors of muSSP solver for releasing the code. Please cite their paper if you find it useful:
```
@inproceedings{wang2019mussp,
  title={muSSP: Efficient Min-cost Flow Algorithm for Multi-object Tracking},
  author={Wang, Congchao and Wang, Yizhi and Wang, Yinxue and Wu, Chiung-Ting and Yu, Guoqiang},
  booktitle={Advances in Neural Information Processing Systems},
  pages={423--432},
  year={2019}
}
```

## Installation
```
export PYTHONPATH="...parent folder.../mot3d:$PYTHONPATH"
```
#### Build muSSP
1. `unzip mussp-master-6cf61b8.zip`
2. `cd ./mot3d/mot3d/solvers/wrappers/muSSP; ./cmake_and_build.sh`

## Usage
Check the examples!
