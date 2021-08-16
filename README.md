## Evaluating Self-supervised Learning Algorithms on the PathMnist Dataset
- The goal of this project is to use 10% of the labeled training data to finetune the linear layer of a classifier, while using only the images of the remaining 90% to pretrain some backbone network. 
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

## Todo:
- Add more SSL algos.
 
