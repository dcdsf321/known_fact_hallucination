import torch
from tuned_lens.plotting import PredictionTrajectory
from jaxtyping import Float
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
import numpy as np
from fancy_einsum import einsum


def get_max_logit_dist(prompt,model,target,device):
    if target.startswith(' ' or '0' or '1' or '2' or '3' or '4' or '5' or '6' or '7' or '8' or '9'):
        first_tok_idx = 2
    else:
        first_tok_idx = 1
    llama_tokens = model.to_tokens(prompt,prepend_bos=True)
    ans_str = model.to_str_tokens(target)[first_tok_idx]
    ans_tokens = model.to_tokens(target)
    answer_toks_wo_head = torch.tensor([ans_tokens[0][first_tok_idx]]).to(device)
    # print(ans_str,ans_tokens, answer_toks_wo_head)
    llama_logits, llama_cache = model.run_with_cache(llama_tokens, remove_batch_dim=True)
    accumulated_residual, acc_labels = llama_cache.accumulated_resid(layer=-1, incl_mid=True, return_labels=True,apply_ln=True)
    scaled_residual_stack = llama_cache.apply_ln_to_stack(accumulated_residual, layer = -1, pos_slice=-1)#好像已经norm了就不会变
    unembed_res = model.unembed(model.ln_final(scaled_residual_stack))
    dist = torch.softmax(unembed_res, dim=-1)
    prob_ans = dist[:,:,answer_toks_wo_head[0]]

    _, word_ranks = dist.sort(descending=True)

    word_ranks_ans = torch.zeros(word_ranks.shape[0],word_ranks.shape[1])
    for i in range(word_ranks.shape[0]):
        for j in range(word_ranks.shape[1]):
            word_ranks_ans[i,j] = torch.arange(len(word_ranks[i,j]))[(word_ranks[i,j] == answer_toks_wo_head).cpu()].item()
    
    llama_tokens_str = model.to_str_tokens(prompt)
    
    return prob_ans, word_ranks_ans, llama_tokens_str


def get_max_tuned_dist(lens,prompt,model,target,device):
    if target.startswith(' ' or '0' or '1' or '2' or '3' or '4' or '5' or '6' or '7' or '8' or '9'):
        first_tok_idx = 2
    else:
        first_tok_idx = 1
    llama_tokens = model.to_tokens(prompt,prepend_bos=True)
    ans_str = model.to_str_tokens(target)[first_tok_idx]
    ans_tokens = model.to_tokens(target)
    answer_toks_wo_head = torch.tensor([ans_tokens[0][first_tok_idx]]).to(device)
    logits, cache = model.run_with_cache(llama_tokens)
    predictition_traj_cache = PredictionTrajectory.from_lens_and_cache(
    lens = lens,
    cache = cache,
    model_logits=logits,
    input_ids=llama_tokens,
    )
    dist = torch.tensor(predictition_traj_cache.probs).squeeze()
    prob_ans = dist[:,:,answer_toks_wo_head[0]]

    _, word_ranks = dist.sort(descending=True)
    word_ranks_ans = torch.zeros(word_ranks.shape[0],word_ranks.shape[1])
    for i in range(word_ranks.shape[0]):
        for j in range(word_ranks.shape[1]):
            word_ranks_ans[i,j] = torch.arange(len(word_ranks[i,j]))[(word_ranks[i,j].cpu() == answer_toks_wo_head.cpu())].item()
    
    llama_tokens_str = model.to_str_tokens(prompt)
    del(predictition_traj_cache)
    return prob_ans, word_ranks_ans, llama_tokens_str

def get_neuron_dist(prompt,model,target,device):
    if target.startswith(' ' or '0' or '1' or '2' or '3' or '4' or '5' or '6' or '7' or '8' or '9'):
        first_tok_idx = 2
    else:
        first_tok_idx = 1
    llama_tokens = model.to_tokens(prompt)
    ans_tokens = model.to_tokens(target)
    llama_logits, llama_cache = model.run_with_cache(llama_tokens, remove_batch_dim=True)
    llama_tokens_str = model.to_str_tokens(prompt)
    stack_sub_state = torch.stack([llama_cache["post",l,'mlp'] for l in range(len(model.blocks))]) #layer tok d_subspace
    #model.W_out: layer d_subspace d_model
    #neuron_updates: layer tok d_subspace d_model
    answer_toks_wo_head = torch.tensor([ans_tokens[0][first_tok_idx]]).to(device)
    ans_str = model.to_str_tokens(target)[first_tok_idx]
    answer_residual_directions = model.tokens_to_residual_directions(answer_toks_wo_head)
    max_neu_contrib = torch.zeros([len(model.blocks),len(llama_tokens_str)]).to(device)
    min_neu_contrib = torch.zeros([len(model.blocks),len(llama_tokens_str)]).to(device)
    for l in range(len(model.blocks)):
        for i in range(len(llama_tokens_str)):
            tmp =  model.W_out[l] * torch.unsqueeze(stack_sub_state[l,i], 1) 
            tmp = tmp.to(device) #3000MB
            neuron_answer_contrib = tmp@answer_residual_directions
            max_neu_contrib[l,i] = torch.max(neuron_answer_contrib)
            min_neu_contrib[l,i] = torch.min(neuron_answer_contrib)
    
    return max_neu_contrib, min_neu_contrib,llama_tokens_str, ans_str

def get_neuron_dist_old(prompt,model,target,device):
    if target.startswith(' ' or '0' or '1' or '2' or '3' or '4' or '5' or '6' or '7' or '8' or '9'):
        first_tok_idx = 2
    else:
        first_tok_idx = 1
    llama_tokens = model.to_tokens(prompt)
    ans_tokens = model.to_tokens(target)
    llama_logits, llama_cache = model.run_with_cache(llama_tokens, remove_batch_dim=True)
    llama_tokens_str = model.to_str_tokens(prompt)
    stack_sub_state = torch.stack([llama_cache["post",l,'mlp'] for l in range(len(model.blocks))]) #layer tok d_subspace
    #model.W_out: layer d_subspace d_model
    #neuron_updates: layer tok d_subspace d_model
    print(1)
    neuron_updates = torch.zeros([stack_sub_state.shape[0],len(llama_tokens_str),model.W_out.shape[1],model.W_out.shape[2]])#.to('cuda')
    print(2)
    for l in range(len(model.blocks)):
        for i in range(len(llama_tokens_str)):
            neuron_updates[l,i] =  model.W_out[l] * torch.unsqueeze(stack_sub_state[l,i], 1) 
    print(3)
    answer_toks_wo_head = torch.tensor([ans_tokens[0][first_tok_idx]]).to(device)
    ans_str = model.to_str_tokens(target)[first_tok_idx]
    answer_residual_directions = model.tokens_to_residual_directions(answer_toks_wo_head)
    print(4)
    max_neu_contrib = torch.zeros([len(model.blocks),len(llama_tokens_str)]).to(device)
    min_neu_contrib = torch.zeros([len(model.blocks),len(llama_tokens_str)]).to(device)
    print(5)
    for l in range(len(model.blocks)):
        for i in range(len(llama_tokens_str)):
            tmp = neuron_updates[l,i].to(device)
            neuron_answer_contrib = tmp@answer_residual_directions
            max_neu_contrib[l,i] = torch.max(neuron_answer_contrib)
            min_neu_contrib[l,i] = torch.min(neuron_answer_contrib)
    
    return max_neu_contrib, min_neu_contrib,llama_tokens_str, ans_str

def get_logits_dist(prompt,output,model,target,device,incl_mid = True, tok_w_space = False):
    if tok_w_space:
        first_tok_idx = 0
    else:
        if target.startswith(' ' or '0' or '1' or '2' or '3' or '4' or '5' or '6' or '7' or '8' or '9'):
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
    scaled_residual_stack = llama_cache.apply_ln_to_stack(accumulated_residual, layer = -1, pos_slice=-1)
    unembed_res = model.unembed(model.ln_final(scaled_residual_stack))
    dist = torch.softmax(unembed_res, dim=-1)
    prob_ans = dist[:,:,answer_toks_wo_head[0]]
    
    return prob_ans, llama_tokens_str, ans_str

#access right answer logit lens
def get_tuned_logits_dist(lens,prompt,model,target,device, tok_w_space = False):
    if tok_w_space:
        first_tok_idx = 0
    else:
        if target.startswith(' ' or '0' or '1' or '2' or '3' or '4' or '5' or '6' or '7' or '8' or '9'):
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


def get_ablated_logits_change(model, prompt, target, device, ablate_part = 'mlp'):
    if target.startswith(' ' or '0' or '1' or '2' or '3' or '4' or '5' or '6' or '7' or '8' or '9'):
        first_tok_idx = 2
    else:
        first_tok_idx = 1
    str_tokens = model.to_tokens(prompt,prepend_bos=True)
    llama_tokens_str = model.to_str_tokens(prompt)
    ans_str = model.to_str_tokens(target)[first_tok_idx]
    n_layers = model.cfg.n_layers
    original_logits, cache = model.run_with_cache(str_tokens)
    ans_tokens = model.to_tokens(target)
    answer_toks_wo_head = torch.tensor([ans_tokens[0][first_tok_idx]]).to(device)
    prob_dist = torch.zeros((n_layers,len(str_tokens[0])))
    # for tok_idx in tqdm(range(len(str_tokens[0]))):
    for tok_idx in range(len(str_tokens[0])):
        def head_ablation_hook(
                value: Float[torch.Tensor, "batch pos head_index d_head"],
                hook: HookPoint
            ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
                # print(f"Shape of the value tensor: {value.shape}")
                value[:,tok_idx,:] = 0
                return value
        for layer_idx in range(n_layers):
            ablated_logits = model.run_with_hooks(
                str_tokens, 
                return_type="logits", 
                fwd_hooks=[(
                    utils.get_act_name(f"{ablate_part}_out", layer_idx),
                    head_ablation_hook
                    )]
                )
            dist = torch.softmax(ablated_logits, dim=-1)
            ori_dist = torch.softmax(original_logits,dim=-1)
            prob_ori = ori_dist[:,:,answer_toks_wo_head[0]]
            prob_ans = dist[:,:,answer_toks_wo_head[0]]
            prob_dist[layer_idx,tok_idx] = prob_ori[0,-1] - prob_ans[0,-1]

    return prob_dist, llama_tokens_str, ans_str

def get_acc_ply_res(llama_cache):
    accumulated_residual, acc_labels = llama_cache.accumulated_resid(layer=-1, incl_mid=True, return_labels=True,apply_ln=True)
    per_layer_residual, per_labels = llama_cache.decompose_resid(layer=-1, return_labels=True,apply_ln=True)
    acc_res = {}
    ply_res = {}
    for i in range(len(acc_labels)):
        acc_res[acc_labels[i]] = accumulated_residual[i]
    for i in range(len(per_labels)):
        ply_res[per_labels[i]] = per_layer_residual[i]
    return acc_res, ply_res

from fancy_einsum import einsum
def cal_la(hf_model, model, llama_cache, layer_num, tok_len, d_model):
    acc_res, ply_res = get_acc_ply_res(llama_cache)
    la = torch.zeros([layer_num,tok_len,d_model])#.to('cuda')
    for l in range(layer_num):
        model_W_V = model.blocks[l].attn.W_V.to('cpu')
        model_b_V = model.blocks[l].attn.b_V.to('cpu')
        model_W_O = model.blocks[l].attn.W_O.to('cpu')
        model_b_O = model.blocks[l].attn.b_O.to('cpu')
        # hf_model.model.layers[l].to('cuda')
        hf_model.model.layers[l].to('cpu')
        h = hf_model.model.layers[l].input_layernorm(acc_res[f"{l}_pre"])
        sim_v = einsum(
                    "... d_model, head_idx d_model d_head -> ... head_idx d_head",
                    # llama_cache["resid_pre", l],
                    h,
                    model_W_V,
                    ) + model_b_V
        sim_vo = einsum(
                    "...head_idx d_head, head_idx d_head d_model -> ... head_idx d_model", 
                    sim_v,
                    model_W_O,
                    ) + model_b_O
        # tok_len = llama_cache["attn_scores", l,"attn"].shape[1]
        head_idx = llama_cache["attn_scores", l,"attn"].shape[0]
        d_model = sim_vo.shape[2]
        # A = llama_cache["attn_scores", l,"attn"]
        A = llama_cache["pattern", l, "attn"]
        a = torch.zeros([tok_len,tok_len,d_model])#.to('cuda')
        for i in range(tok_len):
            for j in range(tok_len):
                a[i,j] = torch.sum(A[:,i,j].view(head_idx,1) * sim_vo[j],dim=0)#sim_vo[j]
        # print(a.shape,la.shape)
        for i in range(tok_len):
            la[l,i] = a[-1,i]
        # hf_model.model.layers[l].to('cpu')
    return la

def get_sub_prob_sum(model,sample,prob_ans_list):
    probs_sub_first = []
    probs_sub_last = []
    probs_sub = []
    probs_second_sub_first = []
    probs_second_sub_last = []
    probs_second_sub = []
    probs_middle = []
    probs_further = []
    probs_last = []
    for i in range(len(prob_ans_list)):
        sub_toks = model.to_tokens(sample[i]['subject'], )
        prompt_toks = model.to_tokens(sample[i]['prompt'])
        if (sub_toks[0][1] not in prompt_toks) or (sub_toks[0][-1] not in prompt_toks):
            print(sample[i]['subject'],sample[i]['prompt'],sub_toks,prompt_toks)
            continue
        sub_first_pos = model.get_token_position(sub_toks[0][1],prompt_toks,mode='first', prepend_bos=True)
        sub_last_pos = model.get_token_position(sub_toks[0][-1],prompt_toks,mode='first', prepend_bos=True)
        prob_sub_first = prob_ans_list[i][:,sub_first_pos]
        prob_sub_last = prob_ans_list[i][:,sub_last_pos]
        prob_sub = np.sum(prob_ans_list[i][:,sub_first_pos:sub_last_pos+1],axis=1)/(sub_last_pos-sub_first_pos+1)
        further_tokens = prompt_toks[0][sub_last_pos+1:-1]
        prob_further_tokens = prob_ans_list[i][:,sub_last_pos+1:-1]
        if (sub_toks[0][1] in further_tokens) and (sub_toks[0][-1] in further_tokens):
            sub_sec_first_pos = model.get_token_position(sub_toks[0][1],further_tokens,mode='first', prepend_bos=True)
            sub_sec_last_pos = model.get_token_position(sub_toks[0][-1],further_tokens,mode='first', prepend_bos=True)
            prob_sec_sub_first = prob_further_tokens[:,sub_sec_first_pos]
            prob_sec_sub_last = prob_further_tokens[:,sub_sec_last_pos]
            prob_sec_sub = np.sum(prob_further_tokens[:,sub_sec_first_pos:sub_sec_last_pos+1],axis=1)/(sub_sec_last_pos-sub_sec_first_pos+1)
            probs_second_sub_first.append(prob_sec_sub_first)
            probs_second_sub_last.append(prob_sec_sub_last)
            probs_second_sub.append(prob_sec_sub)
            middel_tokens = further_tokens[:sub_sec_first_pos]
            prob_middle = np.sum(prob_further_tokens[:,:sub_sec_first_pos],axis=1)/len(middel_tokens)
            probs_middle.append(prob_middle)
            further_tokens = further_tokens[sub_sec_last_pos+1:]
            prob_further_tokens = prob_further_tokens[:,sub_sec_last_pos+1:]
        if len(further_tokens) == 0:
            prob_further = np.zeros(prob_further_tokens.shape[0])
        else:
            prob_further = np.sum(prob_further_tokens,axis=1)/len(further_tokens)
        prob_last = prob_ans_list[i][:,-1]
        probs_sub_first.append(prob_sub_first)
        probs_sub_last.append(prob_sub_last)
        probs_sub.append(prob_sub)
        probs_last.append(prob_last)
        probs_further.append(prob_further)
    summed_probs_sub_first = np.sum(probs_sub_first,axis=0) / len(probs_sub_first)
    summed_probs_sub_last = np.sum(probs_sub_last,axis=0) / len(probs_sub_last)
    summed_probs_sub = np.sum(probs_sub,axis=0) / len(probs_sub)
    summed_probs_last = np.sum(probs_last,axis=0) / len(probs_last)
    summed_probs_further = np.sum(probs_further,axis=0) / len(probs_further)
    if len(probs_second_sub_first) > 0:
        summed_probs_second_sub_first = np.sum(probs_second_sub_first,axis=0) / len(probs_second_sub_first)
        summed_probs_second_sub_last = np.sum(probs_second_sub_last,axis=0) / len(probs_second_sub_last)
        summed_probs_second_sub = np.sum(probs_second_sub,axis=0) / len(probs_second_sub)
        summed_probs_middle = np.sum(probs_middle,axis=0) / len(probs_middle)
        return {
            'sub_first': summed_probs_sub_first,
            'sub_last': summed_probs_sub_last,
            'sub': summed_probs_sub,
            'middle': summed_probs_middle,
            'second_sub_first': summed_probs_second_sub_first,
            'second_sub_last': summed_probs_second_sub_last,
            'second_sub': summed_probs_second_sub,
            'further': summed_probs_further,
            'last': summed_probs_last,
        }
    return {
        'sub_first': summed_probs_sub_first,
        'sub_last': summed_probs_sub_last,
        'sub': summed_probs_sub,
        'further': summed_probs_further,
        'last': summed_probs_last,
    }
