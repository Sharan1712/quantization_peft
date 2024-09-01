# A Study of PEFT Methods applied to Quantized Large Language Models
This repository presents implementation of my Master Thesis. This thesis seeks to optimize LLMs by applying PEFT to quantized models, aiming to replicate QLoRA’s success and assess its effectiveness and performance impact across different strategies and LLMs. The motivation behind these experiments is rooted from the the growing need for efficient tuning and deployment of LLMs in resource-constrained environments. It aims to fill the gap in research of combined application by conducting a detailed study on the application of PEFT methods to quantized LLMs.

### Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── Common
    |   ├── data_utils.py  <- Functions required to load and pre-process data required for fine-tuning.
    |   ├── model_utils.py <- Functions required to load LLMs, quantize and add prepare them for PEFT.
    │   └── utils.py       <- Extra util functions for GPU related
    ├── Data
    │   └── mmlu           <- MMLU data for downstream evaluation
    ├── scripts            <- Contains different bash files for easily running different configurations of Quantization and PEFT
    ├── main.py            <- The main script which is used for Finetuning LLMs. The bash files run this script.
    ├── requirements.txt   <- Python packages required for the project.
------------


### Guidelines
1. Create a python virtual environment with version 3.11 and install the prerequisite packages.
```
pip install -r requirements.txt
```
2. Additional Requirments - Huggingface account, permissions to models you want to access. 
3. Optional - Wandb account for logging results.
4. Login to your huggingface using the cli command, you would need an access token which can be created in [here](https://huggingface.co/settings/tokens)
```
huggingface-cli login
```
5. Run any .sh files from the scripts folder or create new .sh files for the configurations you want to run (refer the main.py to decide the different hyperparamters). 
