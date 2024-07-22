# ANIMATE

___
This is  the source code for "ANIMATE: Unsupervised Attributed Graph Anomaly Detection with Masked Graph Transformers"


## Requirements
- Python >= 3.7
- Pytorch >= 1.10
- Numpy >= 1.21.6
- Scipy >= 1.7.3
- scikit-learn >= 0.24.1





## Dataset

We have already put the datasets in folder named `data`. You can also download `Yelp.pt` from Pygod repo. 




## Running
Take Books dataset as an example:

    python train.py --dataset_name Books
    
___
To train and evaluate on other datasets:

    python train.py --dataset_name Disney
    python train.py --dataset_name Reddit
    python train.py --dataset_name Yelp
