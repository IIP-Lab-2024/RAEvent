import json
import os
from torch.utils.data import Dataset
import random


class PairwiseDataset(Dataset):
    def __init__(self, mode, encoding="utf8", *args, **params):
        self.mode = mode

        self.query_path = "/home/fcy/SCR/database/query/query_5fold_att"
        self.cand_path = "/home/fcy/SCR/database/candidates_att"
        self.labels = json.load(open("/home/fcy/SCR/SCR-Experiment/input_data/label/golden_labels.json", 'r'))
        self.data = []
 
        test_file = "0"  # 
        querys = []
        for i in range(5): # 0,1,2,3,4
            if mode == 'train':
                if test_file == str(i): # test_file=0 
                    continue
                else: # train_file=1,2,3,4
                    querys += json.load(open(os.path.join(self.query_path, 'query_%d.json' % i), 'r'))
            else: # test_file=1,2,3,4
                if test_file == str(i):
                    querys = json.load(open(os.path.join(self.query_path, 'query_%d.json' % i), 'r'))
        pos_num = 0
        self.query2posneg = {}
        # 遍历查询数据，并为每个查询构建正负样本对。正样本是与查询相关的候选，而负样本是不相关的候选。
        for query in querys:
            # 拼接
            crime_str = '，'.join(query['crime']) + "。"
            que = crime_str + query["q"]
            # print(que) 
            # 不拼接
            # que = query["q"]
            path = os.path.join(self.cand_path, str(query["ridx"])) #对应查询案例ridx的那个数据集文件夹
            self.query2posneg[str(query["ridx"])] = {"pos": [], "neg": []}
            for fn in os.listdir(path):
                cand = json.load(open(os.path.join(path, fn), "r"))
                # 拼接
                cand1 = cand["ajName"] + cand["ajjbqk"]
                # print(cand1)
                label = int(fn.split('.')[0]) in self.labels[str(query["ridx"])] #例如fn=256.json  label是一个bool值 判断此候选是不是在goden_label里
                # 为每个查询维护一个字典query2posneg，其中包含正样本和负样本的索引列表。
                if label:
                    self.query2posneg[str(query["ridx"])]["pos"].append(len(self.data))
                else:
                    self.query2posneg[str(query["ridx"])]["neg"].append(len(self.data))
            # --------------------------------------------------------------------- #
                # print("query2posneg:",self.query2posneg)
                self.data.append({
                    "query": que,           #Todo crime_str = '，'.join(b['crime'])
                    "cand": cand1, #Todo crime_str = '，'.join(b['ajName']) cand["ajjbqk"]
                    "label": label, 
                    "index": (query["ridx"], fn.split('.')[0]),  # 查询、待查案例索引
                    "query_inputs": query['inputs'],                # added event info // "inputs": "input_ids":[], "event_type_ids": []
                    "cand_inputs": cand['inputs']                   # added event info
                })
                # 在构建数据样本的过程中，会统计正样本的数量，并在最后打印出来。
                pos_num += int(label)
        print(mode, "positive num:", pos_num)

    def __getitem__(self, item):
        pair1 = self.data[item % len(self.data)]
        return (pair1, )

    def __len__(self):
        if self.mode == "train":
            return len(self.data)
        else:
            return len(self.data)
