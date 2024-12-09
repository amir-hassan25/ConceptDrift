## System 

To ensure optimal performance and compatibility, your system should meet the following requirements:
* **Python and Library Versions**:
	* Python: Version 3.10 or higher
	* PyTorch: Version 2.0 or higher
	* Torch Geometric: Version 2.5.2 or higher
	* CUDA: Version 11.8
* **Hardware Compatibility**:
* ConceptDrift has been tested to successfully work under the following conditions:
	* CPU Systems: Single CPU with at least 32GB of RAM.
	* GPU Systems: Nvidia A40, A6000, or A100 GPU for accelerated computation.

## Environment

To ensure ConceptDrift can run properly on your system, please follow the following steps: 

1. To replicate our environment with all the necessary packages, please install the packages in `requirements.txt` in a virtual environment (e.g. [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [Pip](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) environment) before running our code.
2. Edit `src/config.ini` to include the file path to the `data` folder. 

## Training ConceptDrift

1. To train ConceptDrift, activate your environment with the necessary packages and go to the `src` folder.
2. Execute `python train.py --dataset {dataset}` to train on the `virology`, `neurology`, or `neurology` dataset. 

Hyperparameters can be adjusted with the following command line arguments:  `--batch_size ` (default is 200), `--max_epochs` (default is 2), and `--lr` (default is 0.0001).
