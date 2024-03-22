import torch
import os
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM,AutoTokenizer
from deney import step
from GPU_memory import GPU
import argparse

def main(args):
    tokenizer_path = model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16, device_map='auto')

    targets = json.load(open(args.path_targrts, 'r'))
    predictions = []
    for i, target in tqdm(list(enumerate(targets))):
        init_token=torch.tensor(tokenizer.encode(target)).to("cuda")
        for i in range(args.epoch):
            temp={}
            init_token, loss = step(model, tokenizer, init_token, batch_size=args.batchsize,topk_semanteme=args.topk_semanteme)
            temp["origin"] = target
            temp["adv"] = tokenizer.decode(init_token)
            temp["loss"] = float(loss)
            print(temp)
            predictions.append(temp)
    if not os.path.exists('result'):
        os.makedirs('result')
    json_path = os.path.join('result', args.path_saved)
    json_file = open(json_path, mode='w',encoding='utf-8')
    list_json=json.dumps(predictions, indent=4,ensure_ascii=False)
    json_file.write(list_json)
    json_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default = None)
    parser.add_argument('--path_saved', type=str, default= None)
    parser.add_argument('--path_targrts', type=str, default= None)
    parser.add_argument('--do_sample', type=bool, default= None)
    parser.add_argument('--temperature', type=float, default= None)
    parser.add_argument('--max_new_tokens', type=int, default= None)
    parser.add_argument('--epoch', type=int, default= None)
    parser.add_argument('--batchsize', type=int, default= None)
    parser.add_argument('--topk_semanteme', type=int, default= None)
    main(parser.parse_args())
