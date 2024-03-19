import gc
import numpy as np
import torch
import math
from sentence_transformers.util import (semantic_search,
                                        dot_score,
                                        normalize_embeddings)
import tensorflow_hub as hub

embed = hub.load("../universal_encoder")

def sample_control(model, init, grad, batch_size,topk_semanteme=10):
    curr_embeds = model.get_input_embeddings()(init.unsqueeze(0))
    curr_embeds = torch.nn.functional.normalize(curr_embeds,dim=2)               # queries

    embedding_matrix = model.get_input_embeddings().weight
    embedding_matrix = normalize_embeddings(embedding_matrix)      # corpus

    top_indices=torch.zeros(init.shape[0],topk_semanteme,dtype=init.dtype).to("cuda")
    for i in range(init.shape[0]):
        temp=[]                 
        query=curr_embeds[0][i]                 #query
        corpus=embedding_matrix                 #corpus
        hits = semantic_search(query, corpus, 
                            top_k=topk_semanteme+1,
                            score_function=dot_score)
        for hit in hits:
            for dic in hit[1:]:## don't choose same token as before
                temp.append(dic["corpus_id"])
        top_indices[i]=torch.tensor(temp)

    original_control_toks = init.repeat(batch_size, 1)

    new_token_pos = torch.arange(      
        1, 
        len(init), 
        (len(init)-1) / batch_size,      ## this place has to change as well
        device=grad.device
    ).type(torch.int64)

    new_token_val = torch.gather(
        top_indices[new_token_pos], 1, 
        torch.randint(0, topk_semanteme, (batch_size, 1),device=grad.device)
    )

    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

    return new_control_toks

def step(model,tokenizer,init,batch_size=1024 ,topk_semanteme=10):

    main_device = model.device
    # Aggregate gradients.universal needs to add all gradient.

    with torch.no_grad():
        control_cand = sample_control(model,init,batch_size,topk_semanteme)

    # Search
    loss = torch.zeros( batch_size).to(main_device)
    with torch.no_grad():
        question="summary the meaning of following text:"
        qusetion_list=tokenizer.encode(question).to("cuda")
        full_input=torch.cat((qusetion_list.unsqueeze(0),init.unsqueeze(0)[:,1:]),1).squeeze(0)
        gen_tokens = model.generate(
            full_input,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=100
        )
        original_meaning = tokenizer.batch_decode(gen_tokens)[0]
        for j, cand in enumerate(control_cand):
            question="summary the meaning of following text:"
            qusetion_list=tokenizer.encode(question).to("cuda")
            full_input=torch.cat((qusetion_list.unsqueeze(0),cand.unsqueeze(0)),1).squeeze(0)
            gen_tokens = model.generate(
                full_input,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=100
            )
            gen_text = tokenizer.batch_decode(gen_tokens)[0][len(question):]
            cos_similarity=torch.cosine_similarity(torch.tensor(np.array(embed(gen_text))),torch.tensor(np.array(embed(original_meaning))), dim=1)
            final_similarity=1-torch.acos(cos_similarity)/math.pi
            loss[j]=final_similarity
        min_idx = loss.argmin()
        next_control, cand_loss= control_cand[min_idx], loss[min_idx]

    del  loss ; gc.collect()
    print(next_control,flush=True)
    print(cand_loss,flush=True)
    return next_control,cand_loss