#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/23 6:03 PM
# @Author  : Charles He
# @File    : generate_test.py
# @Software: PyCharm


from transformers import MarianMTModel, MarianTokenizer


def main():
    src_text = [
        '>>zh_HK<< this is a sentence in english that we want to translate to french',
        '>>zh_HK<< This should go to portuguese',
        '>>zh_HK<< And this to Spanish'
    ]
    pre_train = "Helsinki-NLP/opus-mt-en-zh"

    tokenizer = MarianTokenizer.from_pretrained(pre_train)
    model = MarianMTModel.from_pretrained(pre_train)
    model()
    translated = model.generate(**tokenizer.prepare_seq2seq_batch(src_text))
    tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    print(tgt_text)


if __name__ == '__main__':
    main()