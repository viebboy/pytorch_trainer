## Trainer Class for Pytorch
This module implements the training and evaluation functionalities for pytorch model. It has a modular design, which facilitates efficient extension through inheritance. The module `pytorch_trainer` implements two classes: `ClassifierTrainer`, which is used for classification problems (trained with cross entropy loss), and `RegressionTrainer`, which is used for regression problems (trained with mean squared error). Some of the features include:

- training progress bar, printing, tensorboard
- proper validation + testing: best model selected via validation set (if available), results measured on test set
- checkpoint after *k* epoch

