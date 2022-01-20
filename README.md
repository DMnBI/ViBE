## ViBE: a hierarchical BERT model to identify viruses using metagenome sequencing data

This repository includes the implementation of 'ViBE: a hierarchical BERT model to identify viruses using metagenome sequencing data'. Please cite our paper if you use our models. Fill free to report any issue for maintenance of our model.

## Citation
under review  
doi:

## Overview

Here is a brief overview of ViBE's classification workflow.

<img src="https://user-images.githubusercontent.com/15358085/150098730-b4397cbe-46de-4420-a7b1-6865f566c3c9.jpg">

1. Input sequenced reads are going to convert tabular format input  
   (see 3. Data processing section for details)
2. Domain-level classification will be performed.  
   Each query will be classified as **B**acteria, **P**hage, **D**NA viruses, **R**NA viruses.  
   You can consider queries with a low score as *unclassified*  
   (**NOTE** we suggest a score cutoff as 0.9)
3. Order-level classification will be performed for queries classified as viruses.
4. [Optional] Family-, Genus-, Species-level classification could be performed if you fine-tune the pre-trained model for your specific task.

## 1. Setup
We strongly recommend you to use python virtual environment with [Anaconda](https://www.anaconda.com)/[Miniconda](https://docs.conda.io/en/latest/miniconda.html). Moreover, this model works in practical time on GPU/TPU machine. PLEASE use one or more NVIDIA GPUs (It works without errors using just the CPU, but it will take an astronomical amount of time). The details of the machine/environment we used are as follows:

* NVIDIA A100 with 40GB graphic memory
* CUDA 11.0
* python 3.6.13
* pytorch 1.10.0
* [transformers 4.11.3 - huggingface](https://huggingface.co/docs/transformers/index)
* [datasets 1.15.1 - huggingface](https://huggingface.co/docs/datasets/)

Please adjust per\_device\_batch\_size and gradient\_accumulation\_steps according to the specifications of the machine you are using.  
  
Target batch size = (No. of devices) * (per\_device\_batch\_size) * (gradient\_accumulation\_steps)  
e.g. batch size 2,560 = 2 devices * 32 samples/device * 40 steps

### 1.1 Build environment
```
conda update conda (optional)
conda create -n vibe python=3.6
conda activate vibe
```

### 1.2 Install requirements
```
conda install -c huggingface transformers datasets
conda install -c pytorch pytorch torchvision cudatoolkit
conda install scikit-learn
```
Please ensure to install cudatoolkit compatible with you CUDA version. 

### 1.3 Trouble shooting (Optional)
Make sure that the following two import statements work properly

```
from transformers.activations import ACT2FN
from transformers.models.bert.tokenization_bert import BertTokenizer
```

If you get an error in the first statement, check the transformers version.  
If you get an error in the second statement, install tokenizers like this:

```
conda intall -c huggingface tokenizers=0.10.1
```

## 2. How to use ViBE

Vibe consists of **THREE** main functions: **pre-train**, **fine-tune**, **predict**  
There is a main wrapper script `vibe` in the `src` directory

```
vibe {pre-train, fine-tune, predict} [options]
```
you can find details of required/optional parameters for each function with -h option.

```
vibe {pre-train, fine-tune, predict} -h
```

**\* NOTE.** ViBE does NOT require that data be placed in the directory where ViBE is installed. Just pass the location of your data as a proper parameter.

### 2.1 Pre-train
Although a pre-trained model is given in the `models/pre-trained` directory by default, you can re-train using your data.

```
export WORK_DIR={your working directory}
export DATA_DIR=$WORK_DIR/data/pre-train
export CACHE_DIR=$DATA_DIR/cached
export TRAIN_FILE=$DATA_DIR/train.paired.csv
export DEV_FILE=$DATA_DIR/dev.paired.csv
export OUTPUT_DIR=$WORK_DIR/models/my_pre-trained
export CONFIG_FILE=$WORK_DIR/src/configs/ViBE-config-4

vibe pre-train \
    --gpus 0,1 \
    --train_file $TRAIN_FILE \
    --validation_file $DEV_FILE \
    --output_dir $OUTPUT_DIR \
    --config $CONFIG_FILE \
    --overwrite_output_dir \
    --cache_dir $CACHE_DIR \
    --max_seq_length 512 \
    --num_workers 20 \
    --mlm_probability 0.0375 \
    --gradient_accumulation_steps 40 \
    --per_device_batch_size 32 \
    --max_steps 100000 \
    --eval_steps 500 \
    --warmup_ratio 0.2 \
    --learning_rate 4e-4 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6
```

### 2.2 Fine-tune
Fine-tuned models used in our publication are given in the `models/fine-tuned` directory by default.
You can fine-tune the pre-trained model using your data for any specific task.

```
export WORK_DIR={your working directory}
export PRETRAINED_MODEL=$WORK_DIR/models/pre-trained
export DATA_DIR=$WORK_DIR/data/fine-tune
export CACHE_DIR=$DATA_DIR/cached
export TRAIN_FILE=$DATA_DIR/250bp/BPDR.train.paired.csv
export DEV_FILE=$DATA_DIR/250bp/BPDR.dev.paired.csv
export OUTPUT_DIR=$WORK_DIR/models/my_BPDR.250bp

vibe fine-tune \
    --gpus 0,1 \
    --pre-trained_model $PRETRAINED_MODEL \
    --train_file $TRAIN_FILE \
    --validation_file $DEV_FILE \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --cache_dir $CACHE_DIR \
    --max_seq_length 504 \
    --num_workers 20 \
    --num_train_epochs 4 \
    --eval_steps 1000 \
    --per_device_batch_size 32 \
    --warmup_ratio 0.25 \
    --learning_rate 3e-5
```

### 2.3 Predict
You can simply make classification using the fine-tuned model. ViBE makes two output files for each prediction task: `.txt` and `.npy` files. The `.txt` file includes classification results for each query. It has three columns: query id, label, score. The `.npy` file includes the output vector of the classification layer for each query.

```
export WORK_DIR={your working directory}
export FINETUNED_MODEL=$WORK_DIR/models/BPDR.250bp
export DATA_DIR=$WORK_DIR/data/sample
export CACHE_DIR=$DATA_DIR/cached
export SAMPLE_FILE=$DATA_DIR/sample.paired.csv
export OUTPUT_DIR=$WORK_DIR/preds
export OUTPUT_PREFIX=sample

vibe predict \
    --gpus 0
    --model $FINETUNED_MODEL \
    --sample_file $SAMPLE_FILE \
    --output_dir $OUTPUT_DIR \
    --output_prefix $OUTPUT_PREFIX \
    --cache_dir $CACHE_DIR
    --per_device_batch_size 500 \
    --max_seq_length 504 \
    --num_workers 20 \
    --remove_label // when you have 'label' column in your data
```

## 3. Data processing

We provide a number of scripts for data processing. The scripts are placed in the `scritps` directory.

### 3.1 seq2kmer_doc.py

This script converts input sequences into a `.csv` file that is an input format of ViBE. The required columns are as follow:

Paired-ends reads:

|forward |backward |seqid |
|:-------|:--------|:-------|
|TCCA CCAC CACG ACGA ...|CATG ATGT TGTA GTAT ....| query0|

Single-end reads:

|sequence |seqid |
|:--------|:-----|
|TCCA CCAC CACG ACGA ...| query0|

**NOTE1.** Any additional column does not affect performance of ViBE.  
**NOTE2.** `label` column is additionally required for fine-tuning. You have to add `label` column based on your knowledge for generating fine-tuning dataset.  

**USAGE**

```
seq2kmer_doc.py \
    -i sample_1.fasta \
    -p sample_2.fasta \
    -f fasta \
    -k 4 \
    --min-length 150 \
    --max-length 251 \
    -o sample.paired.csv
```

### 3.2 split_data.py

To perform hierarchical classification, input queries have to be split into separated files based on their classification results. You can simply split queries into multiple files using the provided script. 

**USAGE**

```
split_data.py \
    -i sample.paired.csv \
    -p sample.txt \
    -o ./order-level \
    -c 0.9
```

By default, this script generates output files for all labels. You can generate output files for labels of interest.

```
split_data.py \
    -i sample.paired.csv \
    -p sample.txt \
    -o ./order-level \
    -c 0.9 \
    -t Herpesvirales Zurhausenvirales
```

## 4. Run example data
