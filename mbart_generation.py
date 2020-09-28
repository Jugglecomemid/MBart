#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/23 10:31 AM
# @Author  : Charles He
# @File    : mbart_generation.py
# @Software: PyCharm

import os
import torch
import turibolt as bolt
from transformers import MBartForConditionalGeneration, MBartTokenizer


def translate(sentences_lst, tokenizer, model, num_beams, device):
    model = model.eval().to(device)
    # batch = tokenizer.prepare_seq2seq_batch(sentences_lst)

    input_ids = tokenizer.encode(sentences_lst, return_tensors='pt').to(device)

    outputs = model.generate(input_ids=input_ids, max_length=20, num_beams=num_beams)


    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(bolt.ARTIFACT_DIR, 'MBart_translation.pt')


    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")
    model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-cc25')
    print("loading model")
    model.load_state_dict(torch.load(model_path))
    print("model loaded")
    sentences_lst = "i love you"

    result = translate(sentences_lst, tokenizer, model, 3, device)
    print(result)


if __name__ == '__main__':
    main()