python3 create.py\
  --model_path "../vicuna-7b"\
  --path_saved "./result/vicuna"\
  --path_targrts "./novel.json"\
  --do_sample False\
  --temperature 0\
  --max_new_tokens 200\
  --epoch 10 \
  --batchsize 32 \
  --topk_semanteme 10