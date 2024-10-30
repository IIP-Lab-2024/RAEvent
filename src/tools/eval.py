import logging
import os
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from timeit import default_timer as timer
import json
import torch.nn as nn

logger = logging.getLogger(__name__)


def gen_time_str(t):
    t = int(t)
    minute = t // 60
    second = t % 60
    return '%2d:%02d' % (minute, second)


def output_value(epoch, mode, step, time, loss, end):

    delimiter = " "
    s = ""
    s = s + str(epoch) + " "
    while len(s) < 7:
        s += " "
    s = s + str(mode) + " "
    while len(s) < 14:
        s += " "
    s = s + str(step) + " "
    while len(s) < 25:
        s += " "
    s += str(time)
    while len(s) < 40:
        s += " "
    s += str(loss)
    while len(s) < 48:
        s += " "
    
    s = s.replace(" ", delimiter)
    if not (end is None):
        print(s, end=end)
    else:
        print(s)

def valid(model, dataset, epoch, mode):
    model.eval()
    local_rank = -1
    criterion = nn.CrossEntropyLoss()
    acc_result = None
    total_loss = 0
    cnt = 0
    total_len = len(dataset)
    start_time = timer()
    output_info = ""

    output_time = 1
    step = -1
    more = ""
    if total_len < 10000:
        more = "\t"
 
    res_scores = []
    for step, data in enumerate(dataset):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                data[key] = Variable(data[key].cuda())
        with torch.no_grad():
            logits, pooler_out, attention_weights = model(data, "valid", acc_result)
        loss = criterion(logits, data["labels"])

        acc_result = accuracy(logits, data["labels"], acc_result)
        total_loss += float(loss)

        score = torch.softmax(logits, dim=1)
        cnt += 1
        # data["index"] = [(3817, '22824'), (6909, '172'), ..........]
        res_scores += list(zip(data["index"], score[:, 1].tolist()))
        # res_scores = [((3817, '22824'), 0.016399357467889786), ((6909, '172'), 0.021122360602021217), ......]
        if step % output_time == 0 and local_rank <= 0:
            delta_t = timer() - start_time

            output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                         "%.3lf" % (total_loss / (step + 1)), '\r')
    del data
    # del results

    predictions = {}
    for res in res_scores:
        # print(res[1]) = ((3817, '22824'), 0.016399357467889786)
        if res[0][0] not in predictions:
            predictions[res[0][0]] = []
        predictions[res[0][0]].append((res[0][1], res[1]))
        # {3817:[('22824', 0.016399357467889786), ('28405', 0.2848455309867859), ......], .....}
    for key in predictions:
        predictions[key].sort(key = lambda x:x[1], reverse = True) # 按照概率进行排序。
        predictions[key] = [int(res[0]) for res in predictions[key]] # 将值变成索引（字符串类型转换成int类型） {3817:[22824, ...]}

    result_path = os.path.abspath("/home/fcy/SCR/src/result/test")
    if not os.path.exists(result_path):   
        os.makedirs(result_path, exist_ok=True) # result_path = ./result/EDBERT/test0 
    fout = open(os.path.join(result_path, "%s-test-%d_epoch-%d.json" % ("pairwise", 0, epoch)), "w")
    print(json.dumps(predictions), file = fout)
    fout.close()

    if step == -1:
        logger.error("There is no data given to the model in this epoch, check your data.")
        raise NotImplementedError

    if local_rank <= 0:
        delta_t = timer() - start_time
        # output_info = output_function(acc_result)
        output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
            gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                    "%.3lf" % (total_loss / (step + 1)), None)

    model.train()
    
def accuracy(logit, label, acc_result):
    if acc_result is None:
        acc_result = {'right': 0, 'actual_num': 0, 'pre_num': 0}
    pred = torch.max(logit, dim=1)[1]
    acc_result['pre_num'] += int((pred == 1).sum())
    acc_result['actual_num'] += int((label == 1).shape[0])
    acc_result['right'] += int((pred[label == 1] == 1).sum())
    return acc_result