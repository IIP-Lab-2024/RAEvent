import torch
import torch.nn as nn

from transformers import BertModel, AutoConfig, AutoModelForMaskedLM, AutoModel
from model.model.personalized_bert import EventBertModel


class PairwisePLM1(nn.Module):
    def __init__(self, *args, **params):
        super(PairwisePLM1, self).__init__()

        plm_path = "/hhd2/fan/SCR/Pretrain_model/bert_base_chinese"
        # plm_path = "/hhd2/fan/SCR/Pretrain_model/SAILER"
        #使用event
        print('\nusing EDBERT (Event Detection BERT)')
        self.model = EventBertModel.from_pretrained(plm_path)  
        self.plm_config = AutoConfig.from_pretrained(plm_path)
        self.lfm = 'Longformer' in self.plm_config.architectures[0]

        self.hidden_size = self.plm_config.hidden_size
        self.fc = nn.Linear(self.hidden_size, 2)

    def forward(self, data, mode, acc_result):
        # inputx: (batch, 2, seq_len)
        pair = 1
        batch, seq_len = data["inputx"].shape[0], data["inputx"].shape[2]
        inputx = data["inputx"].view(batch * pair, seq_len)
        mask = data["mask"].view(batch * pair, seq_len)
        segment = data["segment"].view(batch * pair, seq_len)

        #使用event
        event = data['event'].view(batch * pair, seq_len)
        out = self.model(input_ids=inputx, attention_mask=mask, token_type_ids=segment, event_type_ids=event, output_attentions=True)

        attention_weights = out.attentions[-1]
        y = out['pooler_output'].squeeze(1)
        logits = self.fc(y)
        
        return logits, y, attention_weights                  # model.save

