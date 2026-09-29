# m5C-SSNet

Integrating RNA Sequence Semantics and Structural Context for Interpretable 5-Methylcytosine Site Prediction.

m5C-SSNet combines semantic RNA sequence representations learned by a frozen SpliceBERT encoder with structural context derived from base-pair probability matrices (BPPMs). A BPPM-guided GINE encoder and a one-dimensional CNN learn structural representations, while an adaptive gated fusion module integrates sequence and structure information for m5C site prediction.



## Setup

### Requirements

The following package versions reproduce the current `evo2` environment used to train and evaluate m5C-SSNet:

Python==3.12.12

matplotlib==3.10.8

numpy==1.26.4

pytorch-lightning==2.6.1

scikit-learn==1.8.0

torch==2.10.0+cu128

transformers==5.2.0

ViennaRNA==2.7.2

The current environment uses an NVIDIA GPU with CUDA 12.8 PyTorch wheels. ViennaRNA provides the `RNAfold` executable used to generate BPPM caches. Existing valid BPPM caches can be reused without running RNAfold again.

### Create an environment

We highly recommend using a virtual environment for the installation of m5C-SSNet and its dependencies. A virtual environment can be created and (de)activated as follows by using [conda](https://conda.io/docs/):

```sh
# create
conda create -n m5C-SSNet_Env python=3.12.12
# activate
conda activate m5C-SSNet_Env
```

### Install m5C-SSNet

After creating and activating the environment, download and install m5C-SSNet from GitHub:

```sh
git clone https://github.com/Uyiad519/m5C-SSNet.git
cd m5C-SSNet
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The `requirements.txt` file uses the official PyTorch CUDA 12.8 wheel index. For a CPU-only environment, install the CPU build of PyTorch first and then install the remaining dependencies as appropriate for your platform.



## Get Started

### Pre-trained model download

The pre-trained language model used in m5C-SSNet is [**SpliceBERT.1024nt**](https://zenodo.org/record/7995778/files/models.tar.gz?download=1). Download and place the model files in the following directory:

```text
models_folder/SpliceBERT.1024nt/
```

The directory must contain the model configuration, tokenizer files, and `pytorch_model.bin`.

### Data preparation

Training and evaluation data should be provided in FASTA format. Each FASTA header must contain `label=0` or `label=1`, for example:

```text
>sample_1 label=1
ACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGA
>sample_2 label=0
TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCAT
```

Prediction also accepts unlabeled FASTA records. RNA bases represented by `U` are converted to `T` internally. By default, the structural branch uses a centered 41-nt sequence window.

BPPM caches are stored in `data/bppm_cache`. If a valid cache with the same dataset name and sequence content is available, it is loaded directly. Otherwise, m5C-SSNet uses RNAfold to generate the cache before training or prediction.

### Training

In the `utils/config.py` file, you can adjust the hyperparameters.

> In `utils/config.py`, the main variables are defined as follows:
>
>> ***seed*** controls model initialization. The default value `None` generates and records a new random seed for every run. Set an integer to reproduce a run.
>>
>> ***batch_size*** is the training batch size.
>>
>> ***num_epochs*** is the maximum number of training epochs.
>>
>> ***early_stopping_patience*** is the number of validation epochs allowed without sufficient AUC improvement.
>>
>> ***learning_rate*** is the learning rate for the trainable model components.
>>
>> ***output_dir*** is the directory where logs, checkpoints, configurations, and evaluation results are saved.
>>
>> ***accelerator*** and ***gpu_devices*** specify the training device.
>>
>> ***bppm_top_k*** and ***minimum_pairing_probability*** control BPPM-derived structural edges.

If you do not modify the file, the default parameters will be used. To start training, run:

```bash
python train.py
```

After training, the best validation-AUC and validation-AUPR checkpoints are saved under:

```text
logs/m5C-SSNet/version_<N>/checkpoints/
```

### Prediction

For prediction, provide a trained checkpoint and a FASTA file:

```bash
python predict.py path/to/model.ckpt \
  --test-data-path path/to/yourfile.fasta
```

The script exports per-sample prediction scores as CSV. If FASTA headers contain labels, it also calculates evaluation metrics and saves them as JSON.

To view all prediction options, run:

```bash
python predict.py --help
```




