## ViBE: a hierarchical BERT model to identify viruses using metagenome sequencing data

This repository includes the implementation of 'ViBE: a hierarchical BERT model to identify viruses using metagenome sequencing data'. Please cite our paper if you use our models. Fill free to report any issue for maintenance of our model.

## Citation
If you have used ViBE in your research, please cite the following publication:

Ho-Jin Gwak, Mina Rho, ViBE: a hierarchical BERT model to identify eukaryotic viruses using metagenome sequencing data, Briefings in Bioinformatics, Volume 23, Issue 4, July 2022, bbac204
doi: https://doi.org/10.1093/bib/bbac204

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
If you get an error in the second statement, 

> ImportError: /lib/x86_64-linux-gnu/libm.so.6: version `GLIBC_2.29' not found

reinstall tokenizers like this:

```
conda install -c huggingface tokenizers=0.10.1
```

### 1.4 Install ViBE and download models

The source files and useful scripts are in this repository. The pre-trained and fine-tuned models have been uploaded on **Google Drive** since the size of each model is larger than 100MB. PLEASE make sure to download models after cloning this repository.

```
git clone https://github.com/DMnBI/ViBE.git
cd ViBE
chmod +x src/vibe
```

The `vibe` script in the `src` directory is an executable python script. No additional installation is required.

**Download Models**

Please download the model you need through the link below and save them in the `models` directory. You can also download models using the `download_models.py` script in the `scripts` directory. 

```
chmod u+x scripts/gdown.sh
python scripts/download_models.py -d all -o ./models
```

**Pre-trained model**

* [pre-trained](https://drive.google.com/file/d/100EITt7ZmyjkBl_X1kJ83nfV5jpK_ED1/view?usp=sharing)

**Domain-level classifier**

* [BPDR.150bp](https://drive.google.com/file/d/1nSTwkvfeJ5VTs2__FOIVW9IO-L8iQZid/view?usp=sharing)
* [BPDR.250bp](https://drive.google.com/file/d/1WdawuAiz1E4CYwrtjvd24dNFHUjns9ZZ/view?usp=sharing)

**Order-level classifiers**

* [DNA.150bp](https://drive.google.com/file/d/1HrFwr-VQrUHA9vdUowQtOgTCxb6IBA9u/view?usp=sharing)
* [DNA.250bp](https://drive.google.com/file/d/1C-MMl-tMuTJnEkzTrt7EEIRJKB5OqZha/view?usp=sharing)
* [RNA.150bp](https://drive.google.com/file/d/1JHD146DDftVLmM8yecNxjxR28v8SUtGt/view?usp=sharing)
* [RNA.250bp](https://drive.google.com/file/d/1c_jKpqDE8L7hZOKkiTPai53FNzYVGscp/view?usp=sharing)

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
export DATA_DIR=$WORK_DIR/examples/pre-train
export CACHE_DIR=$DATA_DIR/cached
export TRAIN_FILE=$DATA_DIR/train.csv
export DEV_FILE=$DATA_DIR/dev.csv
export OUTPUT_DIR=$WORK_DIR/models/my_pre-trained
export CONFIG_FILE=$WORK_DIR/src/configs/ViBE-config-4

src/vibe pre-train \
    --gpus 0,1 \
    --train_file $TRAIN_FILE \
    --validation_file $DEV_FILE \
    --output_dir $OUTPUT_DIR \
    --config $CONFIG_FILE \
    --overwrite_output_dir \
    --cache_dir $CACHE_DIR \
    --max_seq_length 512 \
    --num_workers 20 \
    --mlm_probability 0.15 \
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
export DATA_DIR=$WORK_DIR/examples/fine-tune
export CACHE_DIR=$DATA_DIR/cached
export TRAIN_FILE=$DATA_DIR/BPDR.250bp.train.paired.csv
export DEV_FILE=$DATA_DIR/BPDR.250bp.dev.paired.csv
export OUTPUT_DIR=$WORK_DIR/models/my_BPDR.250bp

src/vibe fine-tune \
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
    --eval_steps 80 \
    --per_device_batch_size 32 \
    --warmup_ratio 0.25 \
    --learning_rate 3e-5
```

### 2.3 Predict
You can simply make classification using the fine-tuned model. ViBE makes two output files for each prediction task: `.txt` and `.npy` files. The `.txt` file includes classification results for each query. It has three columns: query id, label, score. The `.npy` file includes the output vector of the classification layer for each query.

```
export WORK_DIR={your working directory}
export FINETUNED_MODEL=$WORK_DIR/models/BPDR.250bp
export DATA_DIR=$WORK_DIR/examples/sample
export CACHE_DIR=$DATA_DIR/cached
export SAMPLE_FILE=$DATA_DIR/SRR14403295.paired.csv
export OUTPUT_DIR=$WORK_DIR/preds
export OUTPUT_PREFIX=SRR14403295

src/vibe predict \
    --gpus 0 \
    --model $FINETUNED_MODEL \
    --sample_file $SAMPLE_FILE \
    --output_dir $OUTPUT_DIR \
    --output_prefix $OUTPUT_PREFIX \
    --cache_dir $CACHE_DIR \
    --per_device_batch_size 500 \
    --max_seq_length 504 \
    --num_workers 20 \
    --remove_label // when you have 'label' column in your data
```

Example of `.txt` file:

|seqid |prediction |score |
|:-----|:----------|-----:|
|query0 |Bacteria |0.99 |
|query1 |RNA_viruses |0.72 |
|query2 |Phage |0.92 |
|query3 |DNA_viruses |0.88 |

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
**NOTE2.** `seqid` column is not required for pre-training.  
**NOTE3.** `label` column is additionally required for fine-tuning. You have to add `label` column based on your knowledge for generating fine-tuning dataset.  

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

### 3.3 download_models.py

Pre-trained models were uploaded on Google Drive. You can download those models through not only the above links but given python script. 

**USAGE**

```
download_models.py \
    -d all \
    -o models
```
Using the above command, all pre-trained models will be downloaded in the `models` directory. You can give relative path of the target directory through `-o` option. Moreover, You can download specific model(s) instead of downloading all models.

```
download_models.py \
    -d BPDR250 DNA250 RNA250 \
    -o models
```


## 4. Run example data

This is an example of ViBE workflow to classify the SARS-CoV-2 dataset. Example sequenced reads are given in the `examples/SARS-CoV-2` directory. 10 000 reads were sub-sampled from the SARS-CoV-2 sample that is reported on the SRA database with accession number SRR14403295.

### 4.1 Data processing

Convert sequenced reads into *k*-mer documents.

```
python scripts/seq2kmer_doc.py \
    -i examples/SARS-CoV-2/SRR14403295_1.10K.fastq \
    -p examples/SARS-CoV-2/SRR14403295_2.10K.fastq \
    -o examples/SARS-CoV-2/SRR14403295.10K.paired.csv \
    -k 4 \
    -f fastq \
    --min-length 150 \
    --max-length 251
```

Among 10 000 reads, 9 968 reads are converted into *k*-mer documents. The rest 32 reads are ignored since their length is shorter than 150bp.

Before run ViBE, make `preds` directory for saving prediction outputs.

```
mkdir examples/SARS-CoV-2/preds
```

### 4.2 Domain-level classification

```
export FINETUNED_MODEL=models/BPDR.250bp
export DATA_DIR=examples/SARS-CoV-2
export CACHE_DIR=$DATA_DIR/cached
export SAMPLE_FILE=$DATA_DIR/SRR14403295.10K.paired.csv
export OUTPUT_DIR=$DATA_DIR/preds
export OUTPUT_PREFIX=SRR14403295.domain

src/vibe predict \
    --gpus 0 \
    --model $FINETUNED_MODEL \
    --sample_file $SAMPLE_FILE \
    --output_dir $OUTPUT_DIR \
    --output_prefix $OUTPUT_PREFIX \
    --cache_dir $CACHE_DIR \
    --per_device_batch_size 500 \
    --max_seq_length 504 \
    --num_workers 20
```

The result files `SRR14403295.domain.txt` and `SRR14403295.domain.npy` will be generated in the `preds` directory. Among 9 968 samples, 9 153 samples are classified as `RNA_viruses` and 8 257 samples exceed confidence score cutoff 0.9.

### 4.3 Order-level classification

Get records classified as `RNA_viruses` with high confidence score over 0.9.

```
python scripts/split_data.py \
    -i examples/SARS-CoV-2/SRR14403295.10K.paired.csv \
    -p examples/SARS-CoV-2/preds/SRR14403295.domain.txt \
    -o examples/SARS-CoV-2/ \
    -c 0.9 \
    -t RNA_viruses
```

The above command generates `RNA_viruses.csv` file in the `examples/SARS-CoV-2` directory.

```
export FINETUNED_MODEL=models/RNA.250bp
export DATA_DIR=examples/SARS-CoV-2
export CACHE_DIR=$DATA_DIR/cached
export SAMPLE_FILE=$DATA_DIR/RNA_viruses.csv
export OUTPUT_DIR=$DATA_DIR/preds
export OUTPUT_PREFIX=SRR14403295.RNA

src/vibe predict \
    --gpus 0 \
    --model $FINETUNED_MODEL \
    --sample_file $SAMPLE_FILE \
    --output_dir $OUTPUT_DIR \
    --output_prefix $OUTPUT_PREFIX \
    --cache_dir $CACHE_DIR \
    --per_device_batch_size 500 \
    --max_seq_length 504 \
    --num_workers 20
```

The result files `SRR14403295.RNA.txt` and `SRR14403295.RNA.npy` will be generated in the `preds` directory. Among 8 257 samples, 7 873 samples are classified as `Nidovirales` and 7 303 samples exceed confidence score cutoff 0.9.
