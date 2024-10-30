import torch
import numpy as np
import random
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from transformers import AutoTokenizer, AutoModel

class ConcatDataset(Dataset):
    """
    Cut case 1 and case 2 and concat them into 512
    """
    def __init__(self, df, config, is_testing=False):
        self.df = df
        self.is_testing = is_testing
        self.max_length = config.max_length
        self.num_labels = config.num_labels
        # self.tokenizer = BertTokenizerFast.from_pretrained(
        #     config.roberta_path,
        #     do_lower_case=True,
        #     add_prefix_space=True,
        #     is_split_into_words=True,
        #     truncation=True
        # )
        self.tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext-large')

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, ix):
        text_q = self.df.iloc[ix]['case1']
        text_a = self.df.iloc[ix]['case2']

        input_ids_q = self.tokenizer.encode(text_q)
        input_ids_a = self.tokenizer.encode(text_a)

        input_ids_q = input_ids_q[:self.max_length-1] + [102]
        input_ids_a = input_ids_a[1:self.max_length] + [102]

        attn_mask = [1] * len(input_ids_q)
        attn_mask += [1] * len(input_ids_a)
        token_type_ids = [0] * len(input_ids_q)
        token_type_ids += [1] * len(input_ids_a)
        input_ids = input_ids_q + input_ids_a

        # PAD
        pad_len = self.max_length * 2 - len(input_ids)
        input_ids += [0] * pad_len
        attn_mask += [0] * pad_len
        token_type_ids += [0] * pad_len

        input_ids, attn_mask, token_type_ids = map(torch.LongTensor,
                                                   [input_ids, attn_mask, token_type_ids])

        encoded_dict = {
            'input_ids1': input_ids,
            'attn_mask1': attn_mask,
            'token_type_ids1': token_type_ids,

        }
        if not self.is_testing:
            labels = list(range(self.num_labels))
            sentiment = self.df.iloc[ix]['label']
            encoded_dict['label'] = torch.tensor(labels.index(sentiment), dtype=torch.long)
        return encoded_dict
    

class MatchDataset(Dataset):
    """
    Cut case 1 and case 2 and concat them into 512
    """
    def __init__(self, df, config, is_testing=False):
        self.df = df
        self.is_testing = is_testing
        self.max_length = config.max_length
        self.num_labels = config.num_labels
        # BertTokenizerFast是一种用于分词和编码文本的工具
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.roberta_path,
            do_lower_case=True, # 将文本转换成小写
            add_prefix_space=True, # 在每个单词前加上空格
            is_split_into_words=True, # 输入的文本已经切分成单词列表
            truncation=True # 如果输入文本超过最大长度，则截断多余部分。
        )

    def __len__(self):
        return self.df.shape[0] # 返回数据集中数据条目（行）的数量

    def __getitem__(self, ix):
        # 首先从DataFrame中获取第ix行上的两列数据
        text_q = self.df.iloc[ix]['case1']
        text_a = self.df.iloc[ix]['case2']
        # 使用之前创建好的tokenizer对这两列文本进行编码
        input_ids_q = self.tokenizer.encode(text_q)
        input_ids_a = self.tokenizer.encode(text_a)
        # 由于BERT模型有一个固定的输入长度限制（max_length），所以需要对编码后的列表进行截断或填充操作。首先，在每个列表末尾加上102这个特殊ID[SEP], 表示序列结束符号
        input_ids_q = input_ids_q[:self.max_length-1] + [102]
        input_ids_a = input_ids_a[:self.max_length-1] + [102]
        # 它们用于表示每个位置上的ID是否是有效的（1）还是填充的（0）。由于初始时没有填充任何0，所以这两个列表都是由1组成。
        attn_mask_q = [1] * len(input_ids_q)
        attn_mask_a = [1] * len(input_ids_a)
        token_type_ids_q = [0] * len(input_ids_q)
        token_type_ids_a = [0] * len(input_ids_a)

        # PAD：然后计算每个列表需要填充多少0来达到最大长度, 接着在每个列表末尾加上相应数量的0
        pad_len_q = self.max_length - len(input_ids_q)
        pad_len_a = self.max_length - len(input_ids_a)
        input_ids_q += [0] * pad_len_q
        input_ids_a += [0] * pad_len_a
        attn_mask_q += [0] * pad_len_q
        attn_mask_a += [0] * pad_len_a
        token_type_ids_q += [0] * pad_len_q
        token_type_ids_a += [0] * pad_len_a

        input_ids_q, attn_mask_q, token_type_ids_q = map(torch.LongTensor, [input_ids_q, attn_mask_q, token_type_ids_q])
        input_ids_a, attn_mask_a, token_type_ids_a = map(torch.LongTensor, [input_ids_a, attn_mask_a, token_type_ids_a])

        encoded_dict = {
            'input_ids1': input_ids_q,
            'attn_mask1': attn_mask_q,
            'token_type_ids1': token_type_ids_q,
            'input_ids2': input_ids_a,
            'attn_mask2': attn_mask_a,
            'token_type_ids2': token_type_ids_a,
        }
        # 如果不是测试模式，则还需要从DataFrame中获取第ix行上的label列数据，并转换成torch.LongTensor类型，并将其添加到字典中。
        if not self.is_testing:
            labels = list(range(self.num_labels))
            sentiment = self.df.iloc[ix]['label']
            encoded_dict['label'] = torch.tensor(labels.index(sentiment), dtype=torch.long)

        return encoded_dict


class DFDataset(Dataset):
    def __init__(self, df, config, is_testing=False):
        self.df = df
        self.is_testing = is_testing
        self.max_length = config.max_length
        self.num_labels = config.num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.roberta_path,
            do_lower_case=True,
            add_prefix_space=True,
            is_split_into_words=True,
            truncation=True
        )

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, ix):
        text_q = self.df.iloc[ix]['case1']

        input_ids = self.tokenizer.encode(text_q)[:-1]
        input_ids = input_ids[:self.max_length-1] + [102]

        attn_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        # PAD
        pad_len = self.max_length - len(input_ids)
        input_ids += [0] * pad_len
        attn_mask += [0] * pad_len
        token_type_ids += [0] * pad_len

        input_ids, attn_mask, token_type_ids = map(torch.LongTensor,
                                                   [input_ids, attn_mask, token_type_ids])

        encoded_dict = {
            'input_ids1': input_ids,
            'attn_mask1': attn_mask,
            'token_type_ids1': token_type_ids,

        }
        if not self.is_testing:
            labels = list(range(self.num_labels))
            sentiment = self.df.iloc[ix]['label']
            encoded_dict['label'] = torch.tensor(labels.index(sentiment), dtype=torch.long)
        return encoded_dict
