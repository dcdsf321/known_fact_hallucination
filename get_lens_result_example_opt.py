import os
import torch
from tqdm import tqdm
import numpy as np
import pickle, json
from transformers import AutoTokenizer, OPTForCausalLM
from transformer_lens import HookedTransformer
from tuned_lens import TunedLens
MODEL_PATH="/root/models/transformers/opt/opt-6.7b"

import random
from datetime import datetime

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
hf_model = OPTForCausalLM.from_pretrained(MODEL_PATH, low_cpu_mem_usage=True)

torch.set_grad_enabled(False)
device = 'cuda:1'
tuned_lens_pth = './tuned-lens/checkpoints/opt-6.7b'

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
model = HookedTransformer.from_pretrained("opt-6.7b", hf_model=hf_model, device=device, n_devices=1, move_to_device=True, fold_ln=False, center_writing_weights=False, center_unembed=False, tokenizer=tokenizer)


lens = TunedLens.from_model_and_pretrained(model, tuned_lens_pth)
lens = lens.to(device)
#append /.. dir
from get_dist import *
from plot_utils import *
from data_utils import *

def get_logits_dist(prompt,output,model,target,device,incl_mid = True, tok_w_space = False):
    if target.startswith(' ') and tok_w_space == False:
        first_tok_idx = 2
    elif target.startswith('0' or '1' or '2' or '3' or '4' or '5' or '6' or '7' or '8' or '9'):
        first_tok_idx = 2
    else:
        first_tok_idx = 1
    llama_tokens = model.to_tokens(prompt)
    ans_tokens = model.to_tokens(target)
    llama_logits, llama_cache = model.run_with_cache(llama_tokens, remove_batch_dim=True)
    llama_tokens_str = model.to_str_tokens(prompt)
    answer_toks_wo_head = torch.tensor([ans_tokens[0][first_tok_idx]]).to(device)
    ans_str = model.to_str_tokens(target)[first_tok_idx]
    accumulated_residual, acc_labels = llama_cache.accumulated_resid(layer=-1, incl_mid=incl_mid, return_labels=True,apply_ln=True)
    scaled_residual_stack = llama_cache.apply_ln_to_stack(accumulated_residual, layer = -1, pos_slice=-1)#好像已经norm了就不会变
    unembed_res = model.unembed(model.ln_final(scaled_residual_stack))
    dist = torch.softmax(unembed_res, dim=-1)
    prob_ans = dist[:,:,answer_toks_wo_head[0]]
    
    return prob_ans, llama_tokens_str, ans_str

def get_tuned_logits_dist(lens,prompt,model,target,device, tok_w_space = False):
    if target.startswith(' ') and tok_w_space == False:
        first_tok_idx = 2
    elif target.startswith('0' or '1' or '2' or '3' or '4' or '5' or '6' or '7' or '8' or '9'):
        first_tok_idx = 2
    else:
        first_tok_idx = 1
    llama_tokens = model.to_tokens(prompt)
    ans_tokens = model.to_tokens(target)
    logits, cache = model.run_with_cache(llama_tokens)
    llama_tokens_str = model.to_str_tokens(prompt)
    answer_toks_wo_head = torch.tensor([ans_tokens[0][first_tok_idx]]).to(device)
    ans_str = model.to_str_tokens(target)[first_tok_idx]
    predictition_traj_cache = PredictionTrajectory.from_lens_and_cache(
    lens = lens,
    cache = cache,
    model_logits=logits,
    input_ids=llama_tokens,
    )
    prob_ans = torch.exp(torch.tensor(predictition_traj_cache.log_probs[0][:,:,answer_toks_wo_head[0]]))
    
    return prob_ans, llama_tokens_str, ans_str


def get_residual_prob(data,model,target='object'):
    max_tok_len = 0
    tuned_ans_list = []
    logit_ans_list = []
    for d in tqdm(data):
        if target == 'object':
            d_target = ' '+d[target]
        elif target == 'output':
            d_target = d[target]
        tuned_prob_ans, llama_tokens_str, ans_str = get_tuned_logits_dist(lens,d['prompt'],model,d_target, device, tok_w_space = True)
        logit_prob_ans, llama_tokens_str, ans_str = get_logits_dist(d['prompt'],d['output'],model,d_target, device,incl_mid = False, tok_w_space = True)
        
        max_tok_len = max(max_tok_len,len(llama_tokens_str))
        tuned_ans_list.append(tuned_prob_ans.cpu().numpy())
        logit_ans_list.append(logit_prob_ans.cpu().numpy())
    return tuned_ans_list, logit_ans_list, max_tok_len
def main():
    data_all = []
    data_r = [] 
    data_z = [] 
    data_pth_list = [
            './model_output/opt-6.7b-hall.jsonl'
        ]
    
    for pth in data_pth_list:
        with open(pth, 'r') as f:
            for line in f:
                line_d = json.loads(line)
                data_all.append(line_d)
    from operator import itemgetter
    from itertools import groupby
    grouped_data = {}
    data_all.sort(key=itemgetter('relation_id'))
    for key, group in groupby(data_all, key=itemgetter('relation_id')):
        grouped_data[str(key)] = list(group)

    file_pth = './results/opt-6.7b/tuned_logits/'
    os.makedirs(file_pth,exist_ok=True)
    for relation_id, relation_data in grouped_data.items():
        print('processing...  ',relation_id)
        data_r, data_z = get_right_zero_from_relation(relation_data)
        sample_size = min(len(data_r),len(data_z))
        sample_r = random.sample(data_r, sample_size)
        sample_z = random.sample(data_z, sample_size)
        pkl_pth = file_pth+relation_id+'.pkl'
        if os.path.exists(pkl_pth):
            print('loading from pkl...')
            with open(pkl_pth, 'rb') as f:
                tuned_ans_list_r, logit_ans_list_r, max_tok_len_r, tuned_ans_list_z, logit_ans_list_z, max_tok_len_z, tuned_ans_list_z_o, logit_ans_list_z_o, max_tok_len_z_o, sample_r, sample_z = pickle.load(f)
        else:
            print('processing...')
            tuned_ans_list_r, logit_ans_list_r, max_tok_len_r = get_residual_prob(sample_r,model)
            tuned_ans_list_z, logit_ans_list_z, max_tok_len_z = get_residual_prob(sample_z,model)
            tuned_ans_list_z_o, logit_ans_list_z_o, max_tok_len_z_o = get_residual_prob(sample_z,model,'output')
            
            with open(pkl_pth, 'wb') as f:
                pickle.dump([tuned_ans_list_r, logit_ans_list_r, max_tok_len_r, tuned_ans_list_z, logit_ans_list_z, max_tok_len_z, tuned_ans_list_z_o, logit_ans_list_z_o, max_tok_len_z_o, sample_r, sample_z], f)
            
if __name__ == "__main__":
    main()