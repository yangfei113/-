import numpy as np
import pickle
import operator
import pandas as pd
import jieba
from language.langconv import *

def Traditional2Simplified(sentence):
    sentence = Converter('zh-hans').convert(sentence)
    return sentence
with open('cmn.txt', 'r', encoding='utf-8') as f:
    lines = f.read().split('\n');
    # 过滤掉 line 中的空元素
    lines = [line for line in lines if line]

source_tokens=[]
target_tokens=[]
for pos, line in enumerate(lines):
    line = line.split('\t')
    e = line[0][:-1] + " " + line[0][-1:]
    c = line[1]
    target_tokens.append(' '.join(e.split(' ')))
    source_tokens.append(' '.join(jieba.lcut(Traditional2Simplified(c).strip(), cut_all=False)))
#     source_tokens.append(' '.join(Traditional2Simplified(c).strip()))


# 生成不同语言的词典
def build_token_dict(token_list):
    token_dict = {
        '<PAD>': 0,
        '<START>': 1,
        '<END>': 2,
    }
    for line in token_list:
        for token in line.split(' '):
            if token not in token_dict:
                token_dict[token]=len(token_dict)
    return token_dict
source_token_dict = build_token_dict(source_tokens)
target_token_dict = build_token_dict(target_tokens)
target_token_dict_inv = {v: k for k, v in target_token_dict.items()}
# 添加特殊符号
encode_tokens = [['<START>'] + tokens.split(' ') + ['<END>'] for tokens in source_tokens]
decode_tokens = [['<START>'] + tokens.split(' ') + ['<END>'] for tokens in target_tokens]
output_tokens = [tokens.split(' ') + ['<END>', '<PAD>'] for tokens in target_tokens]

source_max_len = max(map(len, encode_tokens))
target_max_len = max(map(len, decode_tokens))
encode_tokens = [tokens + ['<PAD>'] * (source_max_len - len(tokens)) for tokens in encode_tokens]
decode_tokens = [tokens + ['<PAD>'] * (target_max_len - len(tokens)) for tokens in decode_tokens]
output_tokens = [tokens + ['<PAD>'] * (target_max_len - len(tokens)) for tokens in output_tokens]
encode_input = [list(map(lambda x: source_token_dict[x], tokens)) for tokens in encode_tokens]
decode_input = [list(map(lambda x: target_token_dict[x], tokens)) for tokens in decode_tokens]
decode_output = [list(map(lambda x: [target_token_dict[x]], tokens)) for tokens in output_tokens]

print(len(encode_input))


import numpy as np
import pickle
import operator
path = 'middle_data/'
with open(path + 'encode_input.pkl', 'wb') as f:
    pickle.dump(encode_input, f, pickle.HIGHEST_PROTOCOL)
with open(path + 'decode_input.pkl', 'wb') as f:
    pickle.dump(decode_input, f, pickle.HIGHEST_PROTOCOL)
with open(path + 'decode_output.pkl', 'wb') as f:
    pickle.dump(decode_output, f, pickle.HIGHEST_PROTOCOL)
with open(path + 'source_token_dict.pkl', 'wb') as f:
    pickle.dump(source_token_dict, f, pickle.HIGHEST_PROTOCOL)
with open(path + 'target_token_dict.pkl', 'wb') as f:
    pickle.dump(target_token_dict, f, pickle.HIGHEST_PROTOCOL)
with open(path + 'source_tokens.pkl', 'wb') as f:
    pickle.dump(source_tokens, f, pickle.HIGHEST_PROTOCOL)