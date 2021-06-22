# Scalable Analytics Project
**Author:** Erick Escobar Gallardo

**Email:** erick.emmanuel.escobar.gallardo@vub.be

**Student ID:** 0573118

**Date:** 23/06/21



The project consist in implementation of an Image Classifier using Deep Neural Networks (CNNs).



## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/).
To install all the requirements, execute the following command:

```bash
pip install requirements.txt
```

## Usage
In order to modify the training and validation constants as well as the directories, it is necessary to modify the 
information in the `utils.py` file. The execution process of the project can be divided in the following phases:
1. Modification of directory paths in `utils.py` accordingly.
2. Execution of `create_dataset.py` to create a smaller sample of the dataset and store it in `./data`.
   This script will split the sample into a training and testing parts, each one with its respective dataframe.
3. Modification of training settings in `utils.py` to set up the CNN architecture, the number of epochs, etc.
4. Execution of training_job.sh using the command `qsub training_job.sh` to start the training of the CNN model
   . The training job will create a checkpoint inside the folder `.\checkpoints`.
5. Execution of `evaluation_job.sh` using the command `qsub evaluation_job.sh` to execute the evaluation pipeline.

    


Connection:
login.hpc.vub.be // vsc10458


```bash
mpiexec -n 2 python -m mpi4py main.py
```
checkpoints: C:\Users\erick\.cache\torch\hub\checkpoints
## Development

### Task 1: A simple neural network
### Task 2: MPI parallelism
### Task 3: Pipelining
### Task 4: Balancing the training
### Task 5: Deep Learning

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

Link to GitHub: https://github.com/erick093/MPI_Pytorch
## License
[MIT](https://choosealicense.com/licenses/mit/)