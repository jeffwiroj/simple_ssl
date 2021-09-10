## Evaluating Self-supervised Learning Algorithms on the PathMnist Dataset
- The goal of this project is to pretrain the backbone of a NN using recent SSL algorithms on 90% of the training set and finetuning the network on 10% of the training set.
For this project, I used resnet34 as the backbone network.

- This dataset diverges from the standard evaluating datasets(Imagenet, Cifar, etc.).



## Data Setup
- Create a folder called data and download the pathmnist dataset(see https://medmnist.com/) into this folder. 

## Baseline (supervised_exp folder): Fully Supervised Training
- dataset.py: Creates and returns the pytorch datasets (train, val, test).
- tune.py: Performs a basic hyperparameter sweep for learning rate and weight decay
- train.py: Trains a NN using the best config found in tune.out
- eval.ipynb: Evaluates on the test set 

## SimSiam (simsiam_exp folder): Pretrains the backbone classifier using SimSiam
- dataset.py: Creates and returns the pytorch datasets (train, unlabel, val, test).
- pretrain.py: Pretrains the backbone using SimSiam on the unlabel dataset
- run_linear.py: Finetunes only the linear layer of the resnet and evaluates the finetuned resnet on the test-set.

## Results:
| Pretrain Method | Test Accuracy (%)|
| --------------- | ----------------- |
| None | 73.62 |
| SimSiam | 77.52 |
| Barlow Twin | 78.2 |

## Todo:
- Add more SSL algos.
 
