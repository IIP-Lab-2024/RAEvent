import torch
import torch.nn as nn

from transformers import BertModel, AutoConfig, AutoModel
from model.model.personalized_bert import RAModel


class PairwisePLM(nn.Module):
    def __init__(self, *args, **params):
        super(PairwisePLM, self).__init__()

        # plm_path = "/hhd2/fan/SCR/Pretrain_model/bert_base_chinese"
        plm_path = "/home/fcy/SCR/Pretrain_model/bert_base_chinese"

        self.model = RAModel.from_pretrained(plm_path)
 
        self.plm_config = AutoConfig.from_pretrained(plm_path, trust_remote_code=True)
        # self.lfm = 'Longformer' in self.plm_config.architectures[0]

        self.hidden_size = self.plm_config.hidden_size
        # self.classifier = nn.Sequential(
        #         nn.Linear(self.hidden_size, self.hidden_size),
        #         nn.LeakyReLU(),
        #         # nn.Dropout(0.5),
        #         nn.Linear(self.hidden_size, self.hidden_size//2),
        #         nn.LeakyReLU(),
        #         nn.Dropout(0.5),
        #         nn.Linear(self.hidden_size//2, 2),
        #         )
        self.fc = nn.Linear(self.hidden_size, 2)


    def forward(self, data, mode, acc_result):
        # inputx: (batch, 2, seq_len)
        pair = 1
        batch, seq_len = data["inputx"].shape[0], data["inputx"].shape[2]
        # 可视化设置
        # batch, seq_len = 1, 140
        inputx = data["inputx"].view(batch * pair, seq_len)
        mask = data["mask"].view(batch * pair, seq_len)
        segment = data["segment"].view(batch * pair, seq_len)
        event = data['event'].view(batch * pair, seq_len)
        out = self.model(input_ids=inputx, attention_mask=mask, token_type_ids=segment, event_type_ids=event, output_attentions=True)
    
        # print(out.keys())
        
        attention_weights = out.attentions[-1]
        # y = out.attentions[-1]

        y = out['pooler_output'].squeeze(1)

        # logits = out['logits']
        logits = self.fc(y)
        
        
        return logits, y, attention_weights                 # model.save

        # loss = self.criterion(result, data["labels"])
        # acc_result = None
        # acc_result = accuracy(result, data["labels"], acc_result)

        # if mode == "train":
            # return {"loss": loss, "acc_result": acc_result}, result
        # return result
        # else:
        #     score = torch.softmax(result, dim=1)  # batch, 2
        #     # return {"loss": loss, "acc_result": acc_result, "score": score[:, 1].tolist(), "index": data["index"]}, result
        #     return {"loss": "111", "acc_result": "222"}, result


# def accuracy(logit, label, acc_result):
#     if acc_result is None:
#         acc_result = {'right': 0, 'actual_num': 0, 'pre_num': 0}
#     pred = torch.max(logit, dim=1)[1]
#     acc_result['pre_num'] += int((pred == 1).sum())
#     acc_result['actual_num'] += int((label == 1).shape[0])
#     acc_result['right'] += int((pred[label == 1] == 1).sum())
#     return acc_result
