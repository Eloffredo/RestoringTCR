import sys, os
import numpy as np
import pandas as pd
from transformers import BertTokenizer,AutoTokenizer,BertForMaskedLM, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from transformers import BertConfig, BertModel
import torch
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, FillMaskPipeline
from datasets import Dataset
import tempfile
from random import sample
from tqdm import tqdm

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union


# wrap the CustomTrainer over the Huggingface Trainer, adapted from https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py
class CustomTrainer(Trainer):
    
    def __init__(self, *args, custom_p: float = None, **kwargs):
        super().__init__(*args, **kwargs)
        assert custom_p > 0, "Define a positive contribution for modified loss"
        self.custom_contr = custom_p
    
    def compute_loss(self, model, inputs: Any, return_outputs: bool = False):
        """
        
        Compute model loss by mixing the standard masked language objective and the custom left-right masked language objective.

        """

        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        # copy the attention mask to implement the left-right MLM
        mod_attention_mask = inputs.get("attention_mask")
        
        range_tensor = ( torch.arange(mod_attention_mask.shape[1]).unsqueeze(0).expand(mod_attention_mask.shape[0], -1) ).to("cuda")
        
        # add a no_attention_mask to only attend one side of the sequence pair, by 
        no_att = (range_tensor < torch.where(inputs.get('input_ids') == tokenizer.additional_special_tokens_ids[0] )[1].unsqueeze(1)).to("cuda")
        
        # set to zero to prevent attention to the left-right context
        mod_attention_mask[no_att] = 0 
                
        #compute contribution from custom loss
        mod_outputs = model( input_ids = inputs.get('input_ids'), attention_mask = mod_attention_mask, labels = labels)
        
        if self.custom_contr:
            # compute standard MLM loss
            outputs = model( input_ids = inputs.get('input_ids'), attention_mask = inputs.get('attention_mask'), labels = labels)
            total_loss = (1-self.custom_contr)*outputs.loss + self.custom_contr * mod_outputs.loss
            
            return (total_loss, outputs) if return_outputs else total_loss
        
        else:
            total_loss = mod_outputs.loss
            return (total_loss, mod_outputs) if return_outputs else total_loss


        
# load tokenizer. Adapted from https://huggingface.co/wukevin/tcr-bert
tokenizer_checkpoint = "./new_tokenizer/"
tokenizer = AutoTokenizer.from_pretrained( tokenizer_checkpoint )

custom_data_collator = CustomDataCollator(tokenizer, mlm = True)

# instantiate custom BERT-like config
config = BertConfig(num_attention_heads=4, 
                    num_hidden_layers=3,
                    intermediate_size = 256, 
                    max_position_embeddings = 64,
                    vocab_size=27,
                    pad_token_id =  tokenizer.pad_token_id,
                    type_vocab_size=1,
                    )
model = AutoModelForMaskedLM.from_config(config)

# move model to gpu
model = model.to("cuda")
folder_model = './MODEL_FOLDER_SAVE'
saving = True

#model_name = model_checkpoint.split("/")[-1]
batch_size = 256
EPOCHS = 150

training_args = TrainingArguments(
    folder_model,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,

    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    #auto_find_batch_size=True,
    num_train_epochs=EPOCHS,
    warmup_ratio=0.01,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    greater_is_better=False,
    push_to_hub=False,
    logging_steps=1,
    save_total_limit=15
)

# define custom trainer 
trainer = CustomTrainer(
    model,
    training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=custom_data_collator
)

# training routine
trainer.train()

# save final model pt
if saving:
    trainer.save_model(output_dir=folder_model)