# Radiology-Report-Summarization

This repo develop a domain adaption training method for radiology report summarization. It has two parts 
- re-training BART model with Masked Language Masking (MLM) task using medical entities
- fine-tuning the model into text summarization data (radiology reports)

Dataset
-----

For retraining, we used the full radiology reports from [MIMIC-III Clinical Database CareVue subset](https://physionet.org/content/mimic3-carevue/1.4/). This could avoid information leak when we fine-tuning the model using [MIMIC-CXR (MIMIC-IV)](https://physionet.org/content/mimic-cxr/2.0.0/) data.

For fine-tuning, we used the same dataset from [MEDIQA 2021](https://sites.google.com/view/mediqa2021). Please ask the organizer if you want to use them.

You'll need to obtain access to the [MIMIC](https://mimic.mit.edu/docs/gettingstarted/). Note that since we are only using the radiology report text data, you do NOT need to download the entire release. For [MIMIC-III Clinical Database CareVue subset](https://physionet.org/content/mimic3-carevue/1.4/), you only need to download the `NOTEEVENT.csv`. FOR [MIMIC-CXR (MIMIC-IV)](https://physionet.org/content/mimic-cxr/2.0.0/), the only file you'll need to download is the compressed report file (`mimic-cxr-reports.zip`).

How to use
-----
Before running this, you need to make sure the following things are in the correct folder:

```
.
├── data
│   ├── fine_tune
│   │   ├── CXR_test.json
│   │   ├── CXR_val.json
│   │   ├── MEDIQA2021_RRS_Test_Set_Full.json
│   │   └── indiana_dev.json
│   └── retrain
│       ├── MIMIC_test_full.txt
│       ├── MIMIC_train_full.txt
│       └── MIMIC_val_full.txt
├── model
│   ├── RM
│   │   ├── README.md
│   │   ├── config.json
│   │   ├── flax_model.msgpack
│   │   ├── gitattributes.txt
│   │   ├── merges.txt
│   │   ├── model.safetensors
│   │   ├── pytorch_model.bin
│   │   ├── tf_model.h5
│   │   ├── tokenizer.json
│   │   └── vocab.json
│   └── SM
│       ├── phrase_SM
│       │   ├── README.md
│       │   ├── added_tokens.json
│       │   ├── config.json
│       │   ├── flax_model.msgpack
│       │   ├── gitattributes.txt
│       │   ├── merges.txt
│       │   ├── model.safetensors
│       │   ├── pytorch_model.bin
│       │   ├── tf_model.h5
│       │   ├── tokenizer.json
│       │   └── vocab.json
│       └── word_SM
│           ├── README.md
│           ├── added_tokens.json
│           ├── config.json
│           ├── flax_model.msgpack
│           ├── gitattributes.txt
│           ├── merges.txt
│           ├── model.safetensors
│           ├── pytorch_model.bin
│           ├── tf_model.h5
│           ├── tokenizer.json
│           └── vocab.json
├── pipeline
│   ├── MLM
│   │   ├── plot_result.ipynb
│   │   ├── run_mlm_RM.py
│   │   └── run_mlm_SM.py
│   └── fine_tune
│       ├── fine_tune_test.py
│       ├── fine_tune_train.py
│       └── utils.py
├── README.md
├── requirements.txt
```

1. to obatin fine-tune data, see [here](https://github.com/abachaa/MEDIQA2021/tree/main/Task3)
2. to obtain re-train data, please contact me once you have the assess to MIMIC data
3. for the models, you need to download the original model files from [huggingface](https://huggingface.co/facebook/bart-base/tree/main)


Once you have all the files ready, you can run it like this:
```
python MLM/run_mlm_RM.py 
    --model_name_or_path ../model/RM
    --line_by_line 
    --num_train_epochs 20
    --train_file ../data/retrain/MIMIC_test_full.txt 
    --validation_file ../data/retrain/MIMIC_test_full.txt
    --per_device_train_batch_size 4
    --per_device_eval_batch_size 4 
    --do_train 
    --do_eval 
    --output_dir ../checkpoint
```

for fine-tuning, you can use pre-trained models from [huggingface](https://huggingface.co/):

```
python fine_tune/fine_tune_train.py 
    --model_checkpoint facebook/bart-base
    --model_save ../fine_tune_results 
    --epoch_num 20
    --max_dataset_size 20000000
```
and test:
```
python test_parse.py 
    --model_save fine_tune_results 
    --model_checkpoint_path ../fine_tune_results_epoch_2_valid_rouge_22.22_model_weights.bin 
    --model_checkpoint facebook/bart-base
```

or you can use the re-training model from MLM step:
```
python fine_tune/fine_tune_train.py 
    --model_checkpoint ../chekpoint/checkpoint-1000
    --model_save ../fine_tune_results 
    --epoch_num 20
    --max_dataset_size 20000000
```
and test:
```
python test_parse.py 
    --model_save fine_tune_results 
    --model_checkpoint_path ../fine_tune_results_epoch_2_valid_rouge_22.22_model_weights.bin 
    --model_checkpoint ../chekpoint/checkpoint-1000
```

Requirements
-------------
- Python==3.8 
- datasets==2.7.0
- evaluate==0.4.0
- lp2==1.8.48
- numpy==1.23.5
- pytorch_lightning==1.8.0
- rouge==1.0.1
- torch==1.12.0+cu113
- torchmetrics==0.10.3
- tqdm==4.64.1
- transformers==4.25.1

Contact
-------
Jinge Wu: jinge.wu.20@ucl.ac.uk

