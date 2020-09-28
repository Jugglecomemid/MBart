#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/12 2:38 PM
# @Author  : Charles He
# @File    : gpu_available.py
# @Software: PyCharm


import torch
a = torch.cuda.is_available()
print(a)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)