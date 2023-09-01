# Debiasing-Attention-Mechanism-in-Transformer-without-Demographics

This repository contains the implementation of "Debiasing Attention Mechanism in Transformers without Demographics." We have re-formulated the attention mechanism in transformers by taking fairness into account, to create a debiased model. Additionally, this repository includes all the baseline models used for comparison in our study.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Contributing](#contributing)


## Installation

### Prerequisites

- Make sure you have Conda installed. If not, install [Anaconda](https://www.anaconda.com/products/distribution).

### Create Conda Environment

1. Create the conda environment from the `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    ```
2. Activate the conda environment:
    ```bash
    conda activate Env
    ```

## Usage
### CelebA
1. **Download the CelebA dataset:** The CelebA dataset is required to run the experiments. You have two options to prepare the dataset:

    - **Automatic Download:** The `celebA_Ours_L.ipynb` notebook includes code in the first block that will automatically download and unzip the CelebA dataset into the correct directory. If you wish to use this option, you will need to uncomment the code in the first block and then run the first block. After downloading the dataset, remember to comment out the code again to prevent double downloading in future runs.
    - **Manual Download:** If you prefer to download the dataset manually, you can download the CelebA dataset and place it in the directory above the root of this repository. You may download CelebA from this [link](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

2. **Run the Main Notebook:** To execute ERM training, please run ``CelebA_ERM.ipynb``. To run our method, Execute the `celebA_Ours_L.ipynb` notebook. 

3. **Run Comparison Methods:** The comparison methods are located in the `Comparison-methods` directory.

    - **JTT:** The code for JTT is adapted from the [official release](https://github.com/anniesch/jtt). To run this method, use the following commands:
        ```
        python generate_downstream.py --exp_name CelebA_sample_exp --dataset CelebA --n_epochs 10 --lr 1e-4 --weight_decay 1e-4 --method ERM
        bash results/CelebA/CelebA_sample_exp/ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1/job.sh
        python process_training.py --exp_name CelebA_sample_exp --dataset CelebA --folder_name ERM_upweight_0_epochs_50_lr_0.0001_weight_decay_0.0001 --lr 1e-04 --weight_decay 1e-04 --final_epoch 1 --deploy
        sbatch results/CelebA/CelebA_sample_exp/train_downstream_ERM_upweight_0_epochs_50_lr_1e-05_weight_decay_0.1/final_epoch1/JTT_upweight_50_epochs_50_lr_1e-05_weight_decay_0.1/job.sh
        ```
    - **Other Methods:** The other methods in the `Comparison-methods` directory are designed to be one-click-to-run. Simply execute the corresponding notebook.

### NLP task
### Hate-Xplain
1. **Prep dataset:** This repo includes the dataset of Hate-Xplain. If you wish to generate the dataset by yourself, we also provide the code:
```
cd Hate-Xplain/HateXplain-master/Data
run Generate_dataset.ipynb
cd ..
cd ..
run Data_prepartation.ipynb
```
2. **Implementation:** You can run our methods with different configurations in one file:

    1. **Choose the Model:** We provide two models: `bert` and `bert_large`. You can select the model by assigning it to the variable `m` in the first block of the notebook. For example, to use `bert_large`, you would assign `m = "bert_large"`.

    2. **Choose the Fine-Tuning Method:** We provide two fine-tuning methods: fine-tuning all layers or fine-tuning only the appended layer. To fine-tune all layers, assign `freeze = False`. To fine-tune only the appended layer, assign `freeze = True`.

    3. **Choose Whether to Use MoCo Training:** To use MoCo training, assign `moco = True`. To not use MoCo training, assign `moco = False`.

    4. **Run the File:** After configuring the settings, you can run the file.

After completing these steps, the notebook will execute with your chosen configurations.


## Contributing

For comparison methods, we adapted from their official release:

- **JTT:** Please refer to [the official JTT repository](https://github.com/anniesch/jtt).
- **LfF:** Please refer to [the official LfF repository](https://github.com/alinlab/LfF).
- **DRO:** Please refer to [the official DRO worksheet](https://worksheets.codalab.org/worksheets/0x17a501d37bbe49279b0c70ae10813f4c/).
- **ARL:** Please refer to [the official ARL repository](https://github.com/google-research/google-research/tree/master/group_agnostic_fairness).


