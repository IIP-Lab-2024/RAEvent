import logging
import os
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
import shutil
from timeit import default_timer as timer
import torch.nn as nn
from .loss import SupConLoss
from tools.eval import valid, gen_time_str, output_value
from tools.initial import init_test_dataset, init_formatter

logger = logging.getLogger(__name__)

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
def checkpoint(filename, model, optimizer, trained_epoch, global_step):
    model_to_save = model.module if hasattr(model, 'module') else model
    save_params = {
        "model": model_to_save.state_dict(),
        "optimizer_name": "adamw",
        "optimizer": optimizer.state_dict(),
        "trained_epoch": trained_epoch,
        "global_step": global_step
    }
 
    try:
        torch.save(save_params, filename)
    except Exception as e:
        logger.warning("Cannot save models with error %s, continue anyway" % str(e))


def train(parameters, do_test=False):
    # print("parameters: ".format(parameters))
    epoch = 4 # epoch = 5
    output_time = 1
    test_time = 1
    scaler = torch.cuda.amp.GradScaler()
    output_path = os.path.join("./output", "PairwiseLecardSAILER_visual")
    if os.path.exists(output_path):
        logger.warning("Output path exists, check whether need to change a name of model")
    os.makedirs(output_path, exist_ok=True)

    trained_epoch = parameters["trained_epoch"] #+ 1
    model = parameters["model"]
    optimizer = parameters["optimizer"] 
    dataset = parameters["train_dataset"]
    global_step = parameters["global_step"]
    # output_function = parameters["output_function"]

    if do_test:
        init_formatter(["test"])
        test_dataset = init_test_dataset()

    step_size = 1
    gamma = float(1)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    exp_lr_scheduler.step() 
    grad_accumulate = 1

    logger.info("Training start....")

    print("Epoch  Stage  Iterations  Time Usage    Loss  ")

    total_len = len(dataset)
    more = ""
    if total_len < 10000:
        more = "\t"
    for epoch_num in range(trained_epoch, epoch):
        start_time = timer()
        current_epoch = epoch_num
        model.train()
        
        acc_result = None
        total_loss = 0

        output_info = ""
        step = -1
        # fgm = FGM(model)
        for step, data in enumerate(dataset):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):   
                    data[key] = Variable(data[key].cuda())

            optimizer.zero_grad()
            logits, pooled_output, attention_weights = model(data, "train", acc_result)
            # logits, pooler_out, attention_weights = model(data, "train", acc_result)
            # 对比损失
            # sloss = SupConLoss().to('cuda')
            # sloss = sloss(pooled_output, data["labels"])
            # sloss = sloss / len(pooled_output)
            # 原来损失
            # print((data["labels"]).shape)
            # print(data["labels"])
            loss = criterion(logits, data["labels"])
            total_loss += float(loss)
            # loss = 0.2 * loss + 0.8 * sloss
            loss.backward()
            # # 对抗训练
            # fgm.attack()  # 在embedding上添加对抗扰动
            # logits, pooled_output, attention_weights = model(data, "train", acc_result)
            # loss_adv = criterion(logits, data["labels"])
            # loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            # fgm.restore()  # 恢复embedding参数
            
            
            acc_result = accuracy(logits, data["labels"], acc_result)
            

            if (step + 1) % grad_accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            if step % output_time == 0:
                # output_info = output_function(acc_result)

                delta_t = timer() - start_time

                output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                    gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                             "%.3lf" % (total_loss / (step + 1)), '\r')

            global_step += 1
            # if step > 20:
            #     break
        exp_lr_scheduler.step()

        # output_info = output_function(acc_result)
        delta_t = timer() - start_time
        output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
            gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                    "%.3lf" % (total_loss / (step + 1)), None)

        if step == -1:
            logger.error("There is no data given to the model in this epoch, check your data.")
            raise NotImplementedError
        
        # 保存模型文件
        # checkpoint(os.path.join(output_path, "%d.pth" % current_epoch), model, optimizer, current_epoch, global_step)

        if current_epoch % test_time == 0:
            with torch.no_grad():
                # pass
                valid(model, parameters["valid_dataset"], current_epoch, mode="valid")
                if do_test:
                    valid(model, test_dataset, current_epoch, mode="test")

            
def accuracy(logit, label, acc_result):
    if acc_result is None:
        acc_result = {'right': 0, 'actual_num': 0, 'pre_num': 0}
    pred = torch.max(logit, dim=1)[1]
    acc_result['pre_num'] += int((pred == 1).sum())
    acc_result['actual_num'] += int((label == 1).shape[0])
    acc_result['right'] += int((pred[label == 1] == 1).sum())
    return acc_result

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
        # exp_logits = torch.exp(logits)

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