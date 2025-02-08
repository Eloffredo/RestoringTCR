import sys, os
import numpy as np
import pandas as pd
from transformers import BertTokenizer,BertForMaskedLM, DataCollatorForLanguageModeling
import torch
from datasets import Dataset
from typing import Any, List, Union
import torch.nn as nn


parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-d", "--device", default = "cuda", help="Specify the device")
parser.add_argument("-pm", "--probmask", default=0.15, help="Specify the masking probability for sampling")
parser.add_argument("-b", "--beta", default=1.00, help="Specify the beta value of softmax to set sampling stringency ")
parser.add_argument("-dir", "--modeldir", help ="Specify model weights directory")
parser.add_argument("-o", "--outfile", help ="Specify output path for saving data")
args = parser.parse_args()

device = args.device
pm = args.probmask
beta = args.beta
output = args.outfile 

assert device in ["cpu","cuda"]
use_gpu = True if device=="cuda" else False

model_checkpoint = args.modeldir
tokenizer_checkpoint = "./new_tokenizer"

tokenizer = BertTokenizer.from_pretrained( tokenizer_checkpoint )
data_collator = DataCollatorForLanguageModeling( tokenizer )

model = BertForMaskedLM.from_pretrained(model_checkpoint)
model = model.to(device)

df = pd.read_csv('./TraindatasetPos_epi_inlist.csv')
df = df[df['epi'].str.len()<= 15]

MAX_SIZE = 450 ### size over which we discard epitopes.
tmp_list = df.groupby('epi')['epi'].count()<=MAX_SIZE
df = df[df['epi'].isin( tmp_list[tmp_list].index.to_list() )]
df.reset_index(inplace=True, drop=True)

train_sequences = []
for epi, tcr in zip(df['epi'],df['tcr']):
    train_sequences.append(" ".join(list(epi)) + " & " + " ".join(list(tcr)) )

print(f'... loading finetuned model ...')


class Generator:
    
    def __init__(
        self,
        p_mask: float = 0.15,
        use_gpu: bool = True,
        sample_all: bool = False,
        mask_special_tokens: bool = False,
        sample_logits: bool = False,
        beta: float = 1,
        device: str = "cuda"):
        
        self.p_mask = p_mask
        self.use_gpu = use_gpu
        self.sample_all = sample_all
        self.mask_special_tokens = mask_special_tokens
        self.sample_logits = sample_logits
        self.beta = beta
        self.device = device
        
        # mapping dictionary specific for this tokenizer
        self.table_conversion = {36:'',124:'',42:'',63:'',46:''}

    
    @staticmethod
    def insert_whitespace(seq: str) -> str:
        return " ".join(list(seq))

    def single_token_iter(self, model: nn.Module, inputs: Any, masked_inputs: Any) -> List[torch.Tensor, torch.Tensor]:
        inputs = inputs.to(self.device)

        with torch.no_grad():
            logits_new = model(**inputs).logits
            
        # create masking pattern for the inputs
        mask = ((torch.rand(inputs.input_ids.shape) > self.p_mask).type(torch.uint8)).to(self.device)
        
        # avoid masking special tokens ~ very imp.
        if not self.mask_special_tokens:
            mask |=(inputs.input_ids >= torch.min(torch.as_tensor(tokenizer.all_special_ids,device=self.device)) )  

        # compute masked inputs for sampling: mask matrix has 0,1 entries 
        masked_inputs.input_ids = inputs.input_ids * mask + tokenizer.mask_token_id * (1 - mask)

        results = model(**masked_inputs).logits
    
        # sanity checkpoint
        if results.shape != logits_new.shape:
            raise ValueError("Shapes of logits before and after masking do not match")
        
        # sample highest likely logit
        if not self.sample_logits:
            new_msa_tokens = torch.argmax(results, dim=2)
    
        # sample from softmax with a temperature
        else:
            new_msa_tokens = torch.distributions.Categorical(probs = torch.softmax(beta*results,dim=2)).sample()
        
        # parse back original tokens if not masked
        if self.sample_all==False:
            new_msa_tokens = inputs.input_ids * mask + new_msa_tokens * (1 -mask)

        inputs.input_ids = new_msa_tokens
    
        # delete from memory
        del mask, results, new_msa_tokens

        return [inputs,masked_inputs]

    # sample a batch of sequences in parallel
    def generate_batch(self, model: nn.Module, sequences: List[str],iters: int =500 ) -> np.ndarray:
        
        iterated_tokens = tokenizer(sequences, padding=True,return_tensors="pt")
        masked_tokens = tokenizer(sequences, padding=True,return_tensors="pt")
    
        if self.use_gpu:
            iterated_tokens = iterated_tokens.to(self.device)
            masked_tokens = masked_tokens.to(self.device)
            
        # iterate many times over sequences to sample new logits    
        for i in range(iters):
            iterated_tokens, masked_tokens = self.single_token_iter(iterated_tokens, masked_tokens)
    
        # move new sampled sequences to cpu
        new_msa_tokens = (iterated_tokens.to("cpu"))
    
        del iterated_tokens, masked_tokens
        
        # move inputs to cpu, cast and return
        return np.array(new_msa_tokens.input_ids.to("cpu")).astype(np.int8)

    def generation_routine(self, model: nn.Module, sequences: List[str], max_iter: int, Ts: int = 500, reps: int = 100) -> None:
    
        new_msa_seqs = [];
        # split the sequences in batch for saving memory usage
        for step in range(int( len(sequences)/Ts ) -1 ):
            print(f'... starting step: {step} ...' )
    
            new_msa = self.generate_batch(sequences[step*Ts:(step+1)*Ts],iters= 50)
            new_msa_seqs.append([self.insert_whitespace(str(tokenizer.decode(new_msa[k])).replace(' ','').translate(self.table_conversion)) for k in range(Ts)] ) 
            # repeat many times over the same batch
            for _ in range(int(reps)):
                new_msa = self.generate_batch(new_msa_seqs[-1], iters = max_iter)
                new_msa_seqs.append([self.insert_whitespace(str(tokenizer.decode(new_msa[k])).replace(' ','').translate(self.table_conversion)) for k in range(Ts)] ) 

        # create dataframe with generated samples
        df = pd.DataFrame([item for sublist in new_msa_seqs for item in sublist],columns =['Agg.'])
        df = df.drop_duplicates()
        
        # save data to filename
        df.reset_index(inplace=True,drop=True)
        #os.makedirs(f'./Dataset',exist_ok=True)
        df.to_csv(f'./Generated_binders_dataset_inlist_BERT_pmask{pm}_beta{beta:.2f}_newloss_DiscSize{MAX_SIZE}.csv',index=0,mode='w')
    
        return None


model.to(device)
max_iter = 15

generative = Generator(
            p_mask = pm, 
            beta = beta, 
            use_gpu = use_gpu, 
            sample_logits = True, 
            sample_all = False, 
            mask_special_tokens = False,
            device = device)

print(f'... running generation for beta: {beta} ... \n' )
generative.generation_routine(train_sequences, model, max_iter = max_iter, reps = 15)