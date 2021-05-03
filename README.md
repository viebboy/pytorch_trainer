## Trainer Class for Pytorch
Module **pytorch_trainer.py** implements the training and evaluation functionalities for pytorch model. It has a modular design, which facilitates efficient extension through inheritance. The module `pytorch_trainer` implements two classes: `ClassifierTrainer`, which is used for classification problems (trained with cross entropy loss), and `RegressionTrainer`, which is used for regression problems (trained with mean squared error). Some of the features include training progress bar, total training time estimation, tensorboard logging, proper model validation (best weight selected via validation set if available), checkpoints and so on. 


### Documentation 
Detailed description of the interface can be found [here](https://github.com/viebboy/pytorch_trainer/blob/master/interface.md)

### Examples
`train_cifar.py` provides a basic example that uses `pytorch_trainer.ClassifierTrainer` to train CIFAR10/CIFAR100 model  
