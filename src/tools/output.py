import json


# out1
def output_function1(data, *args, **params):
    if data['pre_num'] != 0 and data['actual_num'] != 0:
        pre = data['right'] / data['pre_num']
        recall = data['right'] / data['actual_num']
        if pre + recall == 0:
            f1 = 0
        else:
            f1 = 2 * pre * recall / (pre + recall)
    else:
        pre = 0
        recall = 0
        f1 = 0

    metric = {
            'precision': round(pre, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
        }
    return json.dumps(metric)

def output_function(data, config, *args, **params):
    if data['pre_num'] != 0 and data['actual_num'] != 0:
        pre = data['right'] / data['pre_num']
        recall = data['right'] / data['actual_num']
        if pre + recall == 0:
            f1 = 0
        else:
            f1 = 2 * pre * recall / (pre + recall)
    else:
        pre = 0
        recall = 0
        f1 = 0

    metric = {
            'precision': round(pre, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
        }
    return json.dumps(metric)

def binary_function(data, config, *args, **params):
    if data["total"] == 0:
        metric = {"acc": 0}
    else:
        metric = {"acc": round(data["right"] / data["total"], 4)}
    return json.dumps(metric)
