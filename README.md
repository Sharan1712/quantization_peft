# A Study of PEFT Methods applied to Quantized Large Language Models
This repository presents implementation of my Master Thesis. This thesis explores the integration of Quantization and Parameter-Efficient Fine-Tuning (PEFT) methods in optimizing Large Language Models (LLMs). While Quantization and PEFT have independently advanced the efficiency of LLMs, their combined application remains underexplored. This thesis seeks to optimize LLMs through applying PEFT methods to quantized models, aiming to replicate QLoRA’s (Dettmers et al. 2024) success and assess its effectiveness and performance impact across different PEFT strategies and LLMs. The experiments examine various combinations of these techniques, focusing on their impact on model performance, memory efficiency, and computational overhead. Using the LLaMa series as the base models, experiments were conducted across diverse Natural Language Generation tasks to assess these combinations. The results reveal that certain Quantization and PEFT pairings offer substantial improvements in efficiency without significant loss of performance, highlighting the potential for these methods to enable more practical deployment of large models in resource-constrained environments.

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
conda create -n thesis311 python=3.11
conda activate thesis311
pip install -r requirements.txt
```
2. Additional Requirments - Huggingface account, permissions to models you want to access. 
3. Optional - Wandb account for logging results.
4. Login to your huggingface using the cli command, you would need an access token which can be created in [here](https://huggingface.co/settings/tokens)
```
huggingface-cli login
```
5. Run any .sh files from the scripts folder or create new .sh files for the configurations you want to run (refer the main.py to decide the different hyperparamters). 
