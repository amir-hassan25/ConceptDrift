# ConceptDrift Data and Code
This respository holds the virology, neurology, and immunology datasets along with the implementation of ConceptDrift. 

## Data

Use [Git-LFS](https://git-lfs.com/) to ensure data.zip is downloaded. Please unzip `data.zip` in the project directory. You should have a directory named `data` in the project directory. 

Our temporal dynamic graphs are stored as Torch-Geometric [TemporalData](https://pytorch-geometric.readthedocs.io/en/2.5.0/generated/torch_geometric.data.TemporalData.html) objects. You can find a pickle files holding the datasets along with mappings from the node ids to MeSH Terms in the `data/{dataset}` folders. The biobert embeddings for the terms are also provided in the `data/{dataset}` folders. 


## Environment and Config

1. Please install the packages in `requirements.txt` in a virtual environment (e.g. Conda or Pip environment) before running our code.
2. Edit `src/config.ini` to include the file path to the `data` folder. 

## Training ConceptDrift

1. To train ConceptDrift, activate your environment with the necessary packages and go to the `src` folder.
2. Execute `python train.py --dataset {dataset}` to train on the `virology`, `neurology`, or `neurology` dataset. 

You can modify hyperparameters by exploring the command line arguments in  `src/train.py`.

