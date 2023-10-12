# LLM-driven Dialogue State Tracking
Thank you for your interest in our work, and this is the original implementation of "Towards LLM-driven Dialogue State Tracking".

## Local Setup
```
conda create -n LDST python=3.7
conda activate LDST
pip install -r requirements.txt
```

## Data preparation
The two benchmark Datasets can be downloaded at:
* MultiWOZ: https://github.com/budzianowski/multiwoz
* SGD Dataset: https://github.com/google-research-datasets/dstc8-schema-guided-dialogue

We use the data processing script provided by [DST-as-Prompting](https://github.com/chiahsuan156/DST-as-Prompting) for data pre-processing and post-processing.


## Trainnig (`finetune.py`)
```
WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
--nproc_per_node=4 \
--master_port=1234 \
finetune.py \
--base_model 'decapoda-research/llama-7b-hf' \
--data_path './Data/MultiWOZ2_3_preprocess/train_LLM_few-shot-25percent.json' \
--output_dir './LDST_MULTIWOZ23_few-shot_25percent' \
--num_epochs=2 \
--cutoff_len=512 \
--group_by_length \
--lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
--micro_batch_size=8 \
```
This should take ~30 hours to train on a single Nvidia 3090 GPU. At the end of training, the model's fine-tuned weights will be stored in `$output_dir`.

## Inference ( `generate.py`)
You can load the weights that we have provided directly from the `\checkpoint` folder, and make inference.

Zero-shot setting inference:
```
CUDA_VISIBLE_DEVICES=0 python generate_zero-shot.py \
    --load_8bit \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights './Checkpoint_files/Zero-shot_MultiWOZ2-2_except-attraction-domain' \
    --testfile_name './Data/MULTIWOZ22_preprocess/test_LLM.json' \
    --testfile_idx './Data/MULTIWOZ22_preprocess/test_LLM.idx' \
    --output_file './LDST_result_MULTIWOZ22_zero-shot_attraction-domain/test_LLM_result.txt' \
    --except_domain 'attraction'
```
* --except_domain: indicates the unseen domains during training which is the only different parameter between zero-shot inference and few-shot inference.

Few-shot setting inference:
```
CUDA_VISIBLE_DEVICES=0 python generate_few-shot.py \
    --load_8bit \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights './Checkpoint_files/Few-shot_MultiWOZ2-4_1percent' \
    --testfile_name './Data/MULTIWOZ24_preprocess/test_LLM.json' \
    --testfile_idx './Data/MULTIWOZ24_preprocess/test_LLM.idx' \
    --output_file './LDST_result_MULTIWOZ24_few-shot-1percent/test_LLM_result.txt' 
```

## Checkpoint files
We provide all the pre-training weights in the `Checkpoint_files` folder.

## Example output
**Instruction:** Track the state of the slot <hotel-area> in the input dialogue.

**Input:** [USER] I need to book a hotel in the east that has 4 stars. [SYSTEM] I can help you with that. What is your price range? [domain] hotel, [slot] area, it indicates area or place of the hotel. This slot is categorical and you can only choose from the following available values: centre, east, north, south, west.
If the slot is not mentioned in the dialogue, just return NONE. So the value of slot <hotel-area> is


**LDST Ouput:** East
