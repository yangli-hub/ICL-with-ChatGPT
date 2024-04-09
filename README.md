# ICL with ChatGPT
# Author: Yang Li
#The Corresponding Paper: An empirical study of Multimodal Entity-Based Sentiment Analysis with ChatGPT: Improving in-context learning via entity-aware contrastive learning

https://www.sciencedirect.com/science/article/abs/pii/S0306457324000840

# 1. Data set
1) The orginal datasets are Twitter-15 and Twitter-17. These orginal datasets can be saved to the folder of ./org_data
2) The datasets we use is from https://github.com/YangXiaocui1215/GMP. These are subset of Twitter-15 and Twitter-17. 
Save these datasets to the folder ./subset_data
3)The corresponding images can be downloaded via the following link:
https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view


# 2. Apply BLIP to generate image caption, VQA-entity, and VQA-sentiment for all the images.
1) The BLIP models can refer to the hugging face link:https://huggingface.co/collections/Salesforce/blip-models-65242f40f1491fbf6a9e9472 
2) The outputs can be saved to folders ./output_vqa and ./output_caption

#2. Generate Positive and Negative Instances
code: pos_neg_instance.py
The output is saved to the folder: ./output_pos_neg_instances

# 3. train the SimCSE model
1) input:./output_pos_neg_instances
2) The code of SimCSE can be downloaded from https://github.com/princeton-nlp/SimCSE.
The enviroment and steps to train the SimCSE model is listed in the github page.
The model version we use is princeton-nlp/sup-simcse-roberta-large. Other backbone models can also be used based on different tasks.
3) The trained model should be saved.


# 4. Apply the trained Contrastive Leanring model to generate top N samples for few shot learning
prompt_generation.py

# 5. Call ChatGPT Api and perform fewshot learning with the generated few shot samples

