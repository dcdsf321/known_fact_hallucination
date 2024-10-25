from tqdm import tqdm
import json

def get_right_zero_from_relation(data_todo):
    data_r = []
    data_z = []
    for line_d in tqdm(data_todo):
        for i, p in enumerate(line_d['prompts']):
            if line_d['recall'][i] == 1.0:
                if ('A)' in line_d['output'][i]) or ('A.' in line_d['output'][i]):
                    continue
                cat_output = p+line_d['output'][i]
                if len(line_d['target'].split(' ')) > 1:
                    line_d['target'] = line_d['target'].split(' ')[0]
                new_split = cat_output.split(line_d['target'])
                if len(new_split) <= 1:
                    continue
                new_p = new_split[0]
                cat_index = 0
                for k,s in enumerate(new_split):
                    cat_index = k 
                    if len(new_p) < len(line_d['prompts'][i]):
                        new_p = line_d['target'].join(new_split[:cat_index+2])
                    else:
                        break
                if ' not ' in new_p:
                    continue
                line_d['prompts'][i] = new_p

                if line_d['prompts'][i].endswith(' '):
                    line_d['prompts'][i] = line_d['prompts'][i][:-1]
                line_d['output'][i] = line_d['target']+line_d['target'].join(new_split[cat_index+1:])
                case_dict = {
                    'prompt': line_d['prompts'][i],
                    'output': line_d['output'][i],
                    'object': line_d['target'],
                    'relation_id': line_d['relation_id'],
                    'case_id': line_d['case_id'],
                    'recall': line_d['recall'][i],
                    'subject': line_d['subject'],
                }
                data_r.append(case_dict)
            else:
                case_dict = {
                    'prompt': line_d['prompts'][i],
                    'output': line_d['output'][i],
                    'object': line_d['target'],
                    'relation_id': line_d['relation_id'],
                    'case_id': line_d['case_id'],
                    'recall': line_d['recall'][i],
                    'subject': line_d['subject'],
                }
                data_z.append(case_dict)
                
    return data_r, data_z

def get_right_zero_from_rel_prompt(data_todo):
    data_r = {}
    data_z = {}
    for line_d in tqdm(data_todo):
        for i, p in enumerate(line_d['prompts']):
            if line_d['recall'][i] == 1.0:
                if ('A)' in line_d['output'][i]) or ('A.' in line_d['output'][i]):
                    continue
                cat_output = p+line_d['output'][i]
                if len(line_d['target'].split(' ')) > 1:
                    line_d['target'] = line_d['target'].split(' ')[0]
                new_split = cat_output.split(line_d['target'])
                if len(new_split) <= 1:
                    continue
                new_p = new_split[0]
                cat_index = 0
                for k,s in enumerate(new_split):
                    cat_index = k 
                    if len(new_p) < len(line_d['prompts'][i]):
                        new_p = line_d['target'].join(new_split[:cat_index+2])
                    else:
                        break
                if ' not ' in new_p:
                    continue
                line_d['prompts'][i] = new_p
                if line_d['prompts'][i].endswith(' '):
                    line_d['prompts'][i] = line_d['prompts'][i][:-1]
                line_d['output'][i] = line_d['target']+line_d['target'].join(new_split[cat_index+1:])
                case_dict = {
                    'prompt': line_d['prompts'][i],
                    'output': line_d['output'][i],
                    'object': line_d['target'],
                    'relation_id': line_d['relation_id'],
                    'case_id': line_d['case_id'],
                    'recall': line_d['recall'][i],
                    'subject': line_d['subject'],
                }
                if i not in data_r:
                    data_r[i] = []
                data_r[i].append(case_dict)
            else:
                case_dict = {
                    'prompt': line_d['prompts'][i],
                    'output': line_d['output'][i],
                    'object': line_d['target'],
                    'relation_id': line_d['relation_id'],
                    'case_id': line_d['case_id'],
                    'recall': line_d['recall'][i],
                    'subject': line_d['subject'],
                }
                if i not in data_z:
                    data_z[i] = []
                data_z[i].append(case_dict)
                
    return data_r, data_z