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
We used different pre-defined Pytorch Computer Vision Architectures, among these architectures
are: resnet18, resnet34, alexnet, vgg, squeezenet, densenet, inception.

The PyTorch parallelism is disabled using 'torch.set_num_threads(1)'. For this task a well structured training
model is defined. To reduce training time, we can set the constant DEBUG to True that will take a sample of the original
training dataset and use it to train the selected CNN architecture.

### Task 2: MPI parallelism
In order to distribute the training process, first we scatter the dataset to all the nodes. For this
me use MPI.Scatter to distribute the dataset among all the nodes. The dataset is split equally among all the 
processing nodes.

The distributed training process is done using the method MPI Allreduce that reduces (applies a SUM operation)
to gradients of each process. Each process the averages the sum according to the total number processes.

### Task 3: Pipelining
For the pipelining of the testing procedure. We use a simple approach that pipeline the process of reading an image,
resize the image, preprocesses the image (normalize it) and input the image tensor to the model. This pipeline takes into
account the total number of processes, where the first 3 processes are used for the first 3 task, and the rest of the processes
are in charge of the model prediction part.

**IMPORTANT: Remember to first start the training process for an architecture in order to create a checkpoint that will
be used for the pipeline evaluation process.**
### Task 5: Deep Learning
For this task we used 2 different CNN architectures, each for 10 epochs.
<table>
<thead>
  <tr>
    <th>Model name</th>
    <th>Validation score</th>
    <th>Testing Score</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Resnet 34</td>
    <td>0.626262</td>
    <td>0.6931</td>
  </tr>
  <tr>
    <td>Resnet18</td>
    <td>0.8862</td>
    <td>0.6938</td>
  </tr>
</tbody>
</table>

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

Link to GitHub: https://github.com/erick093/MPI_Pytorch
## License
[MIT](https://choosealicense.com/licenses/mit/)