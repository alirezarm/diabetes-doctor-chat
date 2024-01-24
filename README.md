# Diabetes Doctor Chat
This repo shows how to fine tune LLM (e.g., Mistral-7B) on Mac M1 machine using a dataset of patient conversation with doctors and QLoRA fine-tuning. The dataset is extracted from [HealthCareMagic dataset](https://github.com/Kent0n-Li/ChatDoctor) and contains all conversation related to diabetes. [ChatDoctor](https://github.com/Kent0n-Li/ChatDoctor) contains details on how to create an LLM to act as a medical doctor on the full datasets. 

Follow the following steps to fine-tune `Mistral-7B` and run the model as a [Gradio](https://www.gradio.app/) app. The steps can be generalized to other models and datasets, e.g., different models requires different formatting of the dataset.   

## Clone MLX-Examples Repo

```bash
git clone https://github.com/ml-explore/mlx-examples
cd mlx-examples 
```

## Install Dependencies 

```bash
pip3 install -r requirements.txt
``` 

## Copy Codes 

Move content of the `src` folder from this repo into `ml-explore/mlx-examples/lora`.

## Model Quantization    

Create a quantize version of `Mistral-7B` LLM in the model folder  

```bash
python convert.py --hf-path mistralai/Mistral-7B-v0.1 --mlx-pat model/ -q
```

## Prepare Dataset 

Prepare dataset in the format accepted by `Mistral-7B`. Number of samples used in fine tuning and size of train/test/validation splits can be configured.  

```bash
python3 prepare_data.py
```

## Run Fine-Tuning 

```bash
python3 lora.py --model model/ --data data/ --lora-layers 8 --train --iters 1000 
```
This will takes few hours to run. Three factors impacting the training times are `lora-layers`, `iters` and size of the input datasets.  

The adapter file for the above QLoRA fine-tuning is included in this repo (e.g. `adapters.npz`)


## Run Gradio Chat App

We included two simple Gradio apps to chat with the base LLM model (i.e., `gradio_base_llm.py`) and the fine-tuned model (i.e., `gradio_ft_llm.py`).
Run the app and chat: 
```bash 
python3 gradio_ft_llm.py 
``` 

## ROUGE Score Evaluation

```bash 
python3 rouge_evaluation.py 
``` 