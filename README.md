# Syntactic-MTL

This repository contains the multitask learning model proposed in [Granular Syntax Processing with Multi-Task and Curriculum Learning](https://link.springer.com/article/10.1007/s12559-024-10320-1). 

## Requirements
* pytorch
* flair 0.13.0
* fastnlp 0.7.0

## Usage
To train and test the model on two syntactic tasks, put the non-parallel labeled dataset in the `p_paths`, and the parallel labeled dataset in the `c_paths`, e.g., WSJ dataset for `p_paths` and CoNLL2000 for `c_paths`. Then run the following example script:
```
python run.py --batch_size 20 --lr 0.0001 --cl_size 4 --p_paths p_dataset --c_paths c_dataset
```
Note that the model only works on syntax processing tasks with sequence labeling setting.

## Citation
If you use this knowledge base in your work, please cite the paper - [Granular Syntax Processing with Multi-Task and Curriculum Learning](https://link.springer.com/article/10.1007/s12559-024-10320-1) with the following:
```
@article{zhang2024granular,
  title={Granular Syntax Processing with Multi-Task and Curriculum Learning},
  author={Zhang, Xulang and Mao, Rui and Cambria, Erik},
  journal={Cognitive Computation},
  pages={1--15},
  year={2024},
  publisher={Springer}
}
```
