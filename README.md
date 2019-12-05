# Continual Meta Learning for AAAI-2020


A PyTorch implementation or our paper "Learning from the Past: Continual Meta-Learning with
Bayesian Graph Neural Networks"

(unfinished)


### Requirements
- Python 3.6
- Pytorch 1.1
- TensorboardX


### Datasets
The links of datasets will be released afterwards,
- MiniImageNet (1.1 GB) [Link](https://drive.google.com/open?id=15WuREBvhEbSWo4fTr1r-vMY0C_6QWv4w)
- TieredImageNet (12.9 GB) [Link](https://drive.google.com/open?id=1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07)

The data folder should be organized as,
```
/data
/data/mini-imagenet
/data/tiered-imagenet
```

- Download 'mini_imagenet_train/val/test.pickle', and put them in the path 'tt.arg.dataset_root/mini-imagenet/compacted_dataset/'

- After data preparation, please change the arg.dataset_root in train.py

### Training
The general command for training is,
```bash
python3 train.py
```
Change arguments for different experiments:
- dataset: "mini" / "tiered"
- um_unlabeled: for semi-supervised learning
- meta_batch_size: mini_batch size
- num_layers: GNN's depth
- num_cell: number of hidden states 
- num_ways: N-way
- num_shots: K-shot
- seed: we select 111, 222, 333 for reproducibility

Remember to change dataset_root to suit your own case

The training loss and validation accuracy will be automatically saved in './asset/logs/', which can be visualized with tensorboard.
The model weights will be saved in './asset/checkpoints'

### Evaluation
For testing the trained model, you can use the command as
```
python3 eval.py -test_model "THE_MODEL_NAME"
```

### Citation

Please cite the following paper in your publications if it helps your research

    @article{DBLP:journals/corr/abs-1911-04695,
      author    = {Yadan Luo and
                   Zi Huang and
                   Zheng Zhang and
                   Ziwei Wang and
                   Mahsa Baktashmotlagh and
                   Yang Yang},
      title     = {Learning from the Past: Continual Meta-Learning via Bayesian Graph
                   Modeling},
      journal   = {CoRR},
      volume    = {abs/1911.04695},
      year      = {2019},
      url       = {http://arxiv.org/abs/1911.04695},
      archivePrefix = {arXiv},
      eprint    = {1911.04695},
      timestamp = {Mon, 02 Dec 2019 13:44:01 +0100},
      biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1911-04695},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    
### Acknowledgement

Part of code is built on https://github.com/renmengye/few-shot-ssl-public and https://github.com/khy0809/fewshot-egnn

### Contact

To report issues for this code, please open an issue on the issue tracker.
If you have any further questions, please contact me via lyadanluol@gmail.com