# Data preprocessing step
You first need to download the original dataset and put them in this folder:
* MultiWOZ: https://github.com/budzianowski/multiwoz
* SGD Dataset: https://github.com/google-research-datasets/dstc8-schema-guided-dialogue

## Step 1 - Standard Preprocessing
This step is the same as the processing in [DST-as-Prompting](https://github.com/chiahsuan156/DST-as-Prompting), where he only provides code for processing the multiwoz2.2 dataset, and we additionally provide code for processing the MultiWOZ 2.0 (MultiWOZ20_preprocess.py), MultiWOZ 2.4 (MultiWOZ24_preprocess.py) and SGD dataset (SGD_preprocess.py). 

Noted: for SGD dataset, you also need to run the SGD_preprocess_zero-shot.py to get the testing set for zero-shot experiment.

## Step 2 - Instruction Data Generation
We introduced an additional preprocessing stage known as the "Instruction Data Generation Module," as depicted in Figure 4 in the paper.

* To get the few-shot or full data fine tuning data, just run data_prepare_few-shot_{xxx}.py
* To get the zero-shot training data, just run data_prepare_zero-shot_{xxx}.py

  
