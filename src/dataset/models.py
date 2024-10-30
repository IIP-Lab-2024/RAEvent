import json
import math
import jieba
import torch
import torch.nn as nn

from transformers import BertConfig, BertModel,LongformerConfig, AutoModelForMaskedLM
from transformers import RobertaConfig, AutoTokenizer, AutoModel

class BertclsModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        roberta_config = BertConfig.from_pretrained(config.roberta_path)
        roberta_config.output_hidden_states = True # 表示要输出BERT模型的所有隐藏层状态
        self.roberta = AutoModel.from_pretrained(config.roberta_path, config=roberta_config)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(roberta_config.hidden_size, config.num_labels)
        
        torch.nn.init.normal_(self.classifier.weight, std=0.02)
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, input_ids1, attn_mask1, token_type_ids1):
        pooled_output1 = self.roberta(
            input_ids=input_ids1,
            attention_mask=attn_mask1,
            token_type_ids=token_type_ids1
        )['last_hidden_state']
        pooled_output1 = self.dropout(pooled_output1[:,0,:])
        start_logits = self.classifier(pooled_output1)
        return start_logits,pooled_output1
    
class BertClfModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        roberta_config = BertConfig.from_pretrained(config.roberta_path)
        roberta_config.output_hidden_states = True # 表示要输出BERT模型的所有隐藏层状态
        self.roberta = AutoModel.from_pretrained(config.roberta_path, config=roberta_config)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(roberta_config.hidden_size, config.num_labels)
        
        torch.nn.init.normal_(self.classifier.weight, std=0.02)
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, input_ids1, attn_mask1, token_type_ids1):
        pooled_output1 = self.roberta(
            input_ids=input_ids1,
            attention_mask=attn_mask1,
            token_type_ids=token_type_ids1
        )['last_hidden_state']
        pooled_output1 = self.dropout(pooled_output1[:,0,:])
        start_logits = self.classifier(pooled_output1)
        return start_logits,pooled_output1

class MatchModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        roberta_config = BertConfig.from_pretrained(config.roberta_path)
        roberta_config.output_hidden_states = True # 表示要输出BERT模型的所有隐藏层状态
        self.roberta = BertModel.from_pretrained(config.roberta_path, config=roberta_config)
        self.dropout = nn.Dropout(0.5)

        self.classifier = nn.Linear(roberta_config.hidden_size * 2, config.num_labels)
        
        torch.nn.init.normal_(self.classifier.weight, std=0.02) # 正态分布初始化 ，这是一种常用的参数初始化方法，它会根据给定的均值和标准差生成随机数，并赋值给目标tensor。
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, input_ids1, attn_mask1, token_type_ids1, input_ids2, attn_mask2, token_type_ids2):
        pooled_output1 = self.roberta(
            input_ids=input_ids1,
            attention_mask=attn_mask1,
            token_type_ids=token_type_ids1
        )['last_hidden_state']  # --> [batch_size, sequence_length, hidden_size] 
        
        # 对pooled_output1中每个样本的第一个位置（即[CLS]标记对应的位置）进行随机失活
        pooled_output1 = self.dropout(pooled_output1[:, 0, :]) # --> [batch_size, hidden_size]

        pooled_output2 = self.roberta(
            input_ids=input_ids2,
            attention_mask=attn_mask2,
            token_type_ids=token_type_ids2
        )['last_hidden_state']
        pooled_output2 = self.dropout(pooled_output2[:, 0, :])

        pooled_output = torch.cat([pooled_output1, pooled_output2], dim=1) # concat --> [batch_size, hidden_size * 2] 
        start_logits = self.classifier(pooled_output) # --> [batch_size, num_labels]  start_logits可以看作是两个输入文本之间匹配程度（match score）的预测值。
        return start_logits, pooled_output


class TripleMatchModel(nn.Module): # Siamese Mode

    def __init__(self, config):
        super().__init__()

        roberta_config = BertConfig.from_pretrained(config.roberta_path)
        roberta_config.output_hidden_states = True
        self.roberta = BertModel.from_pretrained(config.roberta_path, config=roberta_config)
        self.dropout = nn.Dropout(0.5)

        self.FFN = nn.Sequential(
            nn.Linear(3*roberta_config.hidden_size, roberta_config.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(roberta_config.hidden_size, roberta_config.hidden_size//2),
            nn.LeakyReLU(),
            nn.Linear(roberta_config.hidden_size//2, config.num_labels),
        )
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, input_ids1, attn_mask1, token_type_ids1, input_ids2, attn_mask2, token_type_ids2):
        pooled_output1 = self.roberta(
            input_ids=input_ids1,
            attention_mask=attn_mask1,
            token_type_ids=token_type_ids1
        ).pooler_output
        pooled_output2 = self.roberta(
            input_ids=input_ids2,
            attention_mask=attn_mask2,
            token_type_ids=token_type_ids2
        ).pooler_output
        # Siamese Mode: 它将两个pooler_output以及它们之间的绝对差值拼接起来（torch.cat），得到一个更大维度的向量pooled_output
        pooled_output = torch.cat([pooled_output1, pooled_output2, torch.abs(pooled_output1-pooled_output2)], dim=1)
        start_logits = self.FFN(pooled_output)
        return start_logits, pooled_output


class NaiveBayes:

    def __init__(self, data_path, stopwords='stopwords.txt', threshold=1):
        """
        data: path to a json file
        """
        self.threshold = threshold
        self.item_cot = 0
        self.w_ctr = [{}, {}]
        self.pos_ctr = [{}, {}]
        self.len_ctr = [{}, {}]
        self.w_list = [[], []]
        self.uq_cot = [0, 0]
        self.data_path = data_path
        self.stopwords = set([word.strip() for word in open(stopwords, 'r', encoding='utf-8').readlines()])
        self.build()

    def build(self):

        with open(self.data_path) as f:
            items = json.load(f)

        self.item_cot = len(items)

        for sent, label, sent_id in items:
            words = [word for word in jieba.lcut(sent) if word not in self.stopwords]
            for word in words:
                self.w_ctr[label].setdefault(word, 0)
                self.w_ctr[label][word] += 1
                self.w_list[label].append(word)
            self.pos_ctr[label].setdefault(sent_id, 0)
            self.pos_ctr[label][sent_id] += 1
            self.len_ctr[label].setdefault(len(words), 0)
            self.len_ctr[label][len(words)] += 1

        self.uq_cot[0] = len(set(self.w_list[0]))
        self.uq_cot[1] = len(set(self.w_list[1]))

        print('model for %s built'%self.data_path)

    def __call__(self, text, sent_id):
        """
            text: Chinese sentence
            return: 0 / 1 as label
        ==============================================
            prob = (1 + w_ctr[label][word]) / (len(w_list[label]) + len(unique_word))
        ==============================================
        """
        words = [word for word in jieba.lcut(text) if word not in self.stopwords]
        probs = [0, 0]

        for i in range(2):
            for word in words:
                deno = len(self.w_list[i]) + sum(self.uq_cot)
                if word in self.w_ctr[i]:
                    probs[i] += math.log((self.w_ctr[i][word] + 1) / deno)
                else:
                    probs[i] += math.log(1 / deno)
            deno = len(self.pos_ctr[i]) + len(self.pos_ctr[0]) + len(self.pos_ctr[1])
            if sent_id in self.pos_ctr[i]:
                probs[i] += math.log((self.pos_ctr[i][sent_id] + 1) / deno)
            else:
                probs[i] += math.log(1 / deno)
            deno = len(self.len_ctr[i]) + len(self.len_ctr[0]) + len(self.len_ctr[1])
            if len(words) in self.len_ctr[i]:
                probs[i] += math.log((self.len_ctr[i][len(words)] + 1) / deno)
            else:
                probs[i] += math.log(1 / deno)
            probs[i] += math.log(len(self.w_list[i]) / self.item_cot)

        if probs[0] - probs[1] > self.threshold:
            return 0
        else:
            return 1

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # epsilon是对抗扰动的大小，emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad) # 计算其梯度范数norm
                if norm != 0: # 如果norm不为零，则根据公式r_at = epsilon * param.grad / norm计算对抗扰动r_at，并将其加到词嵌入参数上。
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'): # restore方法是恢复原始词嵌入参数的函数
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name] # 从self.backup字典中取出原始数据并赋值给param.data。最后清空self.backup字典。
        self.backup = {}


class PGD():
    def __init__(self, model, emb_name='word_embeddings', epsilon=1., alpha=0.3):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {
    }
        self.grad_backup = {
    }
    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {
    }
    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r
    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # revise to translate contrastive loss, where the current sentence itself and its translation is 1.
            mask = torch.eye(int(batch_size / 2), dtype=torch.float32)
            mask = torch.cat([mask, mask], dim=1)
            mask = torch.cat([mask, mask], dim=0)
            mask = mask.to(device)

        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            # print(mask)
        else: # mask is not None:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask + 1e-20
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # DONE: I modified here to prevent nan
        mean_log_prob_pos = ((mask * log_prob).sum(1) + 1e-20) / (mask.sum(1) + 1e-20)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # this would occur nan, I think we can divide then sum
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
