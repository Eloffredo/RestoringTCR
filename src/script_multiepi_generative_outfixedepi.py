import sys, os
import numpy as np
import pandas as pd
from transformers import BertTokenizer,AutoTokenizer,BertForMaskedLM, AutoModelForMaskedLM, DataCollatorForLanguageModeling
import torch
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, FillMaskPipeline
from datasets import Dataset
from accelerate import Accelerator


device = str(sys.argv[1])
pm = float(sys.argv[2])
beta = float(sys.argv[3])

assert device in ["cpu","cuda"]
use_gpu = True if device=="cuda" else False

if not use_gpu:
    accelerator = Accelerator(cpu=True)

model_checkpoint = f"./BERT-multiepi/"
tokenizer_checkpoint = "./tcr-bert"

tokenizer = BertTokenizer.from_pretrained( tokenizer_checkpoint )
data_collator = DataCollatorForLanguageModeling( tokenizer )

model = BertForMaskedLM.from_pretrained(model_checkpoint)
model = model.to("cuda")


df = pd.read_csv('./Temp_data_fornew_epi_fixed.csv')
df = df[df['epi'].str.len()<= 15]

MAX_SIZE = 500 ### size over which we discard epitopes.

train_sequences = []
for epi, tcr in zip(df['epi'],df['tcr']):
    train_sequences.append(" ".join(list(epi)) + " & " + " ".join(list(tcr)) )
    

print(f'... loading finetuned model ...')

def insert_whitespace(seq: str) -> str:
    return " ".join(list(seq))

def single_token_iter(inputs,masked_inputs,p_mask,use_gpu = True,sample_all = False,mask_special_tokens = False,sample_logits = False,beta =1):

    device = "cuda" if use_gpu else "cpu"
    inputs = inputs.to(device)

    with torch.no_grad():
            logits_new = model(**inputs).logits

    mask = ((torch.rand(inputs.input_ids.shape) > p_mask).type(torch.uint8)).to(device)

    if not mask_special_tokens:
        mask|=(inputs.input_ids >= torch.min(torch.as_tensor(tokenizer.all_special_ids,device=device)) )  ### avoid masking special tokens ~ very imp.

    tmp_scheme = inputs.input_ids == 26
    rows, indices = torch.arange(inputs.input_ids.shape[0]), torch.argmax(tmp_scheme.to(torch.long), axis=1)

    add_mask = torch.zeros(size=inputs.input_ids.shape, dtype=torch.uint8).to(device)

    for row, index in zip(rows, indices):
        add_mask[row, :index +1] = 1
        
    mask |= add_mask
    masked_inputs.input_ids = inputs.input_ids * mask + tokenizer.mask_token_id * (1 - mask)

    results = model(**masked_inputs).logits
    if results.shape != logits_new.shape:
        raise ValueError("Shapes of logits before and after masking do not match")

    if not sample_logits:
        new_msa_tokens = torch.argmax(results, dim=2)
    elif sample_logits:
        new_msa_tokens = torch.distributions.Categorical(probs = torch.softmax(beta*results,dim=2)).sample()
    if sample_all==False:
        new_msa_tokens = inputs.input_ids * mask + new_msa_tokens * (1 -mask)

    inputs.input_ids = new_msa_tokens

    del mask, results, new_msa_tokens, tmp_scheme, add_mask, rows, indices

    return [inputs,masked_inputs]


def generate_MSA(sequences,p_mask,iters,use_gpu= True, sample_all = False,mask_special_tokens = False,sample_logits = False,beta =1 ):
    
    iterated_tokens = tokenizer(sequences, padding=True,return_tensors="pt")
    masked_tokens = tokenizer(sequences, padding=True,return_tensors="pt")

    if use_gpu:
        iterated_tokens = iterated_tokens.to("cuda")
        masked_tokens = masked_tokens.to("cuda")
        
    for i in range(iters):
        iterated_tokens,masked_tokens = single_token_iter(iterated_tokens,
                                          masked_tokens,
                                          p_mask= p_mask, 
                                          use_gpu = use_gpu ,
                                          sample_all = sample_all,
                                          mask_special_tokens = mask_special_tokens,
                                          sample_logits = sample_logits,
                                          beta = beta)

    new_msa_tokens = (iterated_tokens.to("cpu"))

    del iterated_tokens, masked_tokens
    return np.array(new_msa_tokens.input_ids.to("cpu")).astype(np.int8)

def generation_routine(sequences, pm,beta ,max_iter, Ts = None, reps= 500, use_gpu = True):
    mytable = {36:'',124:'',42:'',63:'',46:''}

    new_msa_seqs = [];

    new_msa = generate_MSA(sequences,p_mask = pm,iters= 1000,use_gpu = use_gpu,sample_all = False,mask_special_tokens=False,sample_logits=True,beta=beta)
    new_msa_seqs.append([insert_whitespace(str(tokenizer.decode(new_msa[k])).replace(' ','').translate(mytable)) for k in range(Ts)] ) 
    
    for _ in range(int(reps)):
        new_msa = generate_MSA(new_msa_seqs[-1],p_mask = pm,iters= max_iter,use_gpu = use_gpu,sample_all = False,mask_special_tokens=False,sample_logits=True,beta=beta)
        new_msa_seqs.append([insert_whitespace(str(tokenizer.decode(new_msa[k])).replace(' ','').translate(mytable)) for k in range(Ts)] ) 
    
    flat_list = [item for sublist in new_msa_seqs for item in sublist]

    df = pd.DataFrame(flat_list,columns =['Agg.'])
    df = df.drop_duplicates()
    
    df.reset_index(inplace=True,drop=True)
    #os.makedirs(f'./Dataset',exist_ok=True)
    df.to_csv(f'./Generated_binders_control_newoutfixed.csv',index=0,mode='w')

    return None

def generation_routine_lg(sequences, pm,beta ,max_iter, Ts = 50, reps = 100, use_gpu = True):
    mytable = {36:'',124:'',42:'',63:'',46:''}

    new_msa_seqs = [];
    for step in range(int( len(sequences)/Ts ) ):
        print(f'... starting step: {step} ...' )

        new_msa = generate_MSA(sequences[step*Ts:(step+1)*Ts],p_mask = pm,iters= 1000,use_gpu = use_gpu,sample_all = False,mask_special_tokens=True,sample_logits=True,beta=beta)
        new_msa_seqs.append([insert_whitespace(str(tokenizer.decode(new_msa[k])).replace(' ','').translate(mytable)) for k in range(Ts)] ) 
    
        for _ in range(int(reps)):
            new_msa = generate_MSA(new_msa_seqs[-1],p_mask = pm,iters= max_iter,use_gpu = use_gpu,sample_all = False,mask_special_tokens=True,sample_logits=True,beta=beta)
            new_msa_seqs.append([insert_whitespace(str(tokenizer.decode(new_msa[k])).replace(' ','').translate(mytable)) for k in range(Ts)] ) 
    
    flat_list = [item for sublist in new_msa_seqs for item in sublist]
    #epi_list , tcr_list = [], []
    #for flat_seq in flat_list:
    #    if '&' in flat_seq:
    #        epi, tcr = flat_seq.replace(' ','').split('&')
    #        epi_list.append(epi)
    #        tcr_list.append(tcr)
    #    else:
    #        continue

    #df = pd.DataFrame(np.array( [epi_list,tcr_list]).T, columns = ['epi','tcr'] )
    #df['binding'] = 1
    df = pd.DataFrame(flat_list,columns =['Agg.'])

    #dff = df[~df['tcr'].str.contains('<')]
    #dff = dff[~dff['tcr'].str.contains('X')]
    #dff = dff[~dff['tcr'].str.contains('>')]
    #dff = dff[~dff['tcr'].str.contains('U')]
    #
    df = df.drop_duplicates()
    
    df.reset_index(inplace=True,drop=True)
    #os.makedirs(f'./Dataset',exist_ok=True)
    df.to_csv(f'./Generated_binders_control_newoutfixed.csv',index=0,mode='w')

    return None


model.to(device)
max_iter = 15

print(f'... running generation for beta: {beta} ... \n' )
if len(train_sequences) < 500:
    generation_routine(train_sequences[:],pm = pm, beta = beta,max_iter = max_iter,reps = 5, use_gpu = use_gpu)
elif len(train_sequences) >= 500:
    generation_routine_lg(train_sequences[:], pm = pm, beta = beta,max_iter = max_iter,reps = 5, use_gpu = use_gpu)
