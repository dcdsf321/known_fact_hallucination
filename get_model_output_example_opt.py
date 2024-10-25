# Import stuff
import torch
import os
import unicodedata
from IPython.display import HTML
# import circuitsvis as cv

from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)

    
from transformers import AutoTokenizer, OPTForCausalLM
import json

def make_prompt(
    prompt_template: str,
    subject: str
) -> str:
    prompt = prompt_template.format(subject)
    # prompt = models.maybe_prefix_eos(mt, prompt) 可能有用
    return prompt

def normalize(s):
    """Normalize answer."""
    import string, re
    s = unicodedata.normalize("NFD", s)
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text) 
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    s = white_space_fix(remove_articles(remove_punc(lower(s))))
    return s

def compute_score(prediction, truth):

    pred_tokens = normalize(prediction).split()
    truth_tokens = normalize(truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    common_tokens = set(pred_tokens) & set(truth_tokens)
    if len(common_tokens) == 0:
        return 0
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    # return 2 * (prec * rec) / (prec + rec)
    return rec
#load counterfact
data_pth = './triplet_data.json'
with open(data_pth, 'r') as f:
    counterfact_data = json.load(f)

MODEL_PATH="/root/models/transformers/opt/opt-6.7b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
hf_model = OPTForCausalLM.from_pretrained(MODEL_PATH, low_cpu_mem_usage=True)

# Loading on CPU is cheapest memory wise in transformer_lens 

n_devices = 1
print("Using {} devices.".format(n_devices))
model = HookedTransformer.from_pretrained("opt-6.7b", hf_model=hf_model, device='cuda:1', fold_ln=False, center_writing_weights=False, center_unembed=False, tokenizer=tokenizer)


for case in counterfact_data:
    recall_score = []
    outputs = []
    for p in case['prompts']:
        output = model.generate(p, max_new_tokens=10, temperature=0,prepend_bos=True)
        output = output.replace(p,'')
        recall_score.append(compute_score(output, case['target']))
        outputs.append(output)
    flag = False
    for r in recall_score:
        if r != recall_score[0]:
            flag = True
            break
    case.update({'recall':recall_score,'output':outputs})
    with open('./model_output/opt-6.7b.jsonl', 'a+') as f:
        f.write(json.dumps(case)+'\n')
    if flag:
        with open('./model_output/opt-6.7b-hall.jsonl', 'a+', 'a+') as f:
            f.write(json.dumps(case)+'\n')