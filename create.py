import torch
import os
import json
import numpy as np
from tqdm import tqdm
import transformers
from transformers import LlamaTokenizer, AutoModelForCausalLM,AutoTokenizer
from new import step

def main():
    
    tokenizer_path = model_path = f"../vicuna"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)  ## stay as a question
    model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16).to('cuda')   ## stay as a question

    targets = json.load(open(f'./fig.json', 'r'))
    predictions = []
    for i, target in tqdm(list(enumerate(targets))):
        init_token=torch.tensor(tokenizer.encode(target)).to("cuda")
        for i in range(15):
            temp={
            "origin":"",
            "adv":"",
            "loss":0,
            "prob":0
            }
            init_token,loss,prob=step(model,init_token,batch_size=512, topk=5,topk_semanteme=10)
            temp["origin"]=target
            temp["adv"]=tokenizer.decode(init_token)
            temp["loss"]=float(loss)
            temp["prob"]=float(prob)
            #print(temp["prob"])
            predictions.append(temp)
    if not os.path.exists('submission'):
        os.makedirs('submission')
    json_path = os.path.join('submission', 'fig.json')
    json_file = open(json_path, mode='w',encoding='utf-8')
    list_json=json.dumps(predictions, indent=4,ensure_ascii=False)
    json_file.write(list_json)
    json_file.close()
if __name__ == "__main__":
    main()