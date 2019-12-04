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
- MiniImageNet
- TieredImageNet



### Training
The general command for training is,
```bash
python 3 train.py
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


