# https://www.deepspeed.ai/tutorials/flops-profiler/

import torch
from transformers import BertForSequenceClassification, BertTokenizer
from deepspeed.profiling.flops_profiler import get_model_profile


def bert_input_constructor(batch_size, seq_len, tokenizer):
    fake_seq = ""
    for _ in range(seq_len - 2):  # ignore the two special tokens [CLS] and [SEP]
      fake_seq += tokenizer.pad_token
    inputs = tokenizer([fake_seq] * batch_size,
                       padding=True,
                       truncation=True,
                       return_tensors="pt")
    labels = torch.tensor([1] * batch_size)
    inputs = dict(inputs)
    inputs.update({"labels": labels})
    return inputs

model_names = ['mrm8488/bert-tiny-finetuned-squadv2',
               'bert-base-uncased',
               'bert-large-uncased',]
for model_name in model_names:
    print(f"{'+' * 20}{model_name}{'+' * 20}")
    with torch.cuda.device(0):
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name)
        batch_size = 4
        seq_len = 128
        enable_profile = True
        if enable_profile:
          flops, macs, params = get_model_profile(
              model,
              kwargs=bert_input_constructor(batch_size, seq_len, tokenizer),
              print_profile=False,
              detailed=False,
              warm_up=10,
          )
          print(f'Model name: {model_name} flops:{flops}, macs:{macs}, params:{params}')
        else:
          inputs = bert_input_constructor((batch_size, seq_len), tokenizer)
          outputs = model(inputs)
