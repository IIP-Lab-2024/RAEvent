import gc
import json
import warnings
import pandas as pd

from pathlib import Path
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import get_linear_schedule_with_warmup

from utils import *
from models import BertClfModel, SupConLoss
from data_module import ConcatDataset

warnings.filterwarnings('ignore')


# @title GlobalConfig
class GlobalConfig:
    def __init__(self):
        self.seed = 2022
        self.path = Path('./data/')
        self.max_length = 256
        self.num_labels = 3
        self.roberta_path = 'hfl/chinese-roberta-wwm-ext-large'  # @param
        self.num_workers = os.cpu_count()
        self.batch_size = 4
        self.steps_show = 100
        self.accum_steps = 1
        num_epochs = 10  # @param
        self.epochs = num_epochs
        self.warmup_steps = 0
        lr = 5e-6  # @param
        self.lr = lr  # modified from 3e-5
        run_id = "stage1_concat_contrastive"  # @param
        self.offline = True
        self.saved_model_path = run_id
        self.n_splits = 5

def model_train(train_loader, val_loader, model, fold):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 1e-3
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.
        }
    ]
    optimizer = optim.AdamW(optimizer_parameters, lr=config.lr)
    train_steps = (len(train_loader) * config.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.warmup_steps * train_steps),
        num_training_steps=train_steps
    )
    steps = 0
    best_f1 = 0
    model.train()

    for epoch in range(1, config.epochs + 1):
        for batch in train_loader:
            input_ids1,attn_mask1,token_type_ids1,label = batch['input_ids1'], batch['attn_mask1'], batch['token_type_ids1'],batch['label']
            if torch.cuda.is_available():
                input_ids1,attn_mask1,token_type_ids1,label = input_ids1.cuda(),attn_mask1.cuda(),token_type_ids1.cuda(),label.cuda()
            # optimizer.zero_grad()
            logits, pooled_output= model(input_ids1,attn_mask1,token_type_ids1)

            sloss = SupConLoss(contrast_mode='all').to('cuda')
            all_sentence_v = pooled_output.unsqueeze(1)
            sloss = sloss(all_sentence_v, label)
            sloss = sloss / len(all_sentence_v)

            loss = nn.CrossEntropyLoss()
            loss=loss(logits,label)

            loss = 0.999 * loss + 0.001 * sloss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            steps += 1
            logits = torch.max(logits.data, 1)[1].cpu()
            label = label.cpu()
            if steps % config.steps_show == 0:
                f1=f1_score(logits,label,average='macro')
                print('epoch:%d\t\t\tsteps:%d\t\t\tloss:%.6f\t\t\tf1_score:%.4f'%(epoch,steps,loss.item(),f1))
        dev_f1 = dev_eval(val_loader,model)
        print('dev\nf1:%.6f'%(dev_f1))
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            torch.save(model,config.saved_model_path+'/'+str(fold)+'.pth')
            print('save best model\t\tf1:%.6f'%best_f1)


def dev_eval(val_loader,model):
    model.eval()
    logits_list=[]
    label_list=[]
    avg_loss=0
    for batch in val_loader:
        input_ids1, attn_mask1, token_type_ids1,  label = batch['input_ids1'], batch['attn_mask1'], batch['token_type_ids1'],  batch['label']
        if torch.cuda.is_available():
            input_ids1, attn_mask1, token_type_ids1, label = input_ids1.cuda(), attn_mask1.cuda(), token_type_ids1.cuda(), label.cuda()
        with torch.no_grad():
            logits, pooled_ = model(input_ids1, attn_mask1, token_type_ids1)
        loss = nn.CrossEntropyLoss()
        loss=loss(logits,label)
        avg_loss += loss.item()
        logits = torch.max(logits.data, 1)[1].cpu()
        label = label.cpu()
        logits_list.extend(logits)
        label_list.extend(label)
    f1 = f1_score(logits_list,label_list,average='macro')
    model.train()
    return f1


def prepare_dataloader(path, shuffle, is_testing=False):
    with open(path) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df.columns = ['ids', 'case1', 'case2', 'label']
    dataset = ConcatDataset(df, config, is_testing=is_testing)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle)

    return dataloader


config = GlobalConfig()
if not os.path.exists(config.saved_model_path):
  os.mkdir(config.saved_model_path)

seed_everything(config.seed)

for fold in range(5):

    train_loader = prepare_dataloader('first_stage2/train' + str(fold) + '.json', True)
    valid_loader = prepare_dataloader('first_stage2/test' + str(fold) + '.json', False)

    model = BertClfModel(config)
    print(model_train(train_loader, valid_loader, model, fold))

    # clean GPU
    model.to('cpu')
    del train_loader, valid_loader, model
    gc.collect()
    torch.cuda.empty_cache()



# save best model		f1:0.717874
# save best model		f1:0.680567
# save best model		f1:0.667167
# save best model		f1:0.709922
# save best model		f1:0.710585