import os
import re
import math
import torch
import random
import pickle
import datetime
import numpy as np
import torch.nn as nn 
from rouge import *
from bleu import compute_bleu
from torch.nn.modules.batchnorm import _BatchNorm


def rouge_score(references, generated):
    """both are a list of strings"""
    score = rouge(generated, references) #调用 Rouge 库的 rouge 函数计算 Rouge 分数
    rouge_s = {k: (v * 100) for (k, v) in score.items()} #将每个 Rouge 分数乘以 100，以得到百分比形式的分数
    '''
    "rouge_1/f_score": rouge_1_f,
    "rouge_1/r_score": rouge_1_r,
    "rouge_1/p_score": rouge_1_p,
    "rouge_2/f_score": rouge_2_f,
    "rouge_2/r_score": rouge_2_r,
    "rouge_2/p_score": rouge_2_p,
    "rouge_l/f_score": rouge_l_f,
    "rouge_l/r_score": rouge_l_r,
    "rouge_l/p_score": rouge_l_p,
    '''
    return rouge_s


def bleu_score(references, generated, n_gram=4, smooth=False):
    """a list of lists of tokens"""
    formatted_ref = [[ref] for ref in references]
    bleu_s, _, _, _, _, _ = compute_bleu(formatted_ref, generated, n_gram, smooth)
    return bleu_s * 100


def two_seq_same(sa, sb):
    if len(sa) != len(sb):
        return False
    for (wa, wb) in zip(sa, sb):
        if wa != wb:
            return False
    return True


def unique_sentence_percent(sequence_batch):
    unique_seq = []
    for seq in sequence_batch:
        count = 0
        for uni_seq in unique_seq:
            if two_seq_same(seq, uni_seq):
                count += 1
                break
        if count == 0:
            unique_seq.append(seq)

    return len(unique_seq) / len(sequence_batch), len(unique_seq)


def feature_detect(seq_batch, feature_set):
    feature_batch = []
    for ids in seq_batch:
        feature_list = []
        for i in ids:
            if i in feature_set:
                feature_list.append(i)
        feature_batch.append(set(feature_list))

    return feature_batch


def feature_matching_ratio(feature_batch, test_feature):
    count = 0
    for (fea_set, fea) in zip(feature_batch, test_feature):
        if fea in fea_set:
            count += 1

    return count / len(feature_batch)


def feature_coverage_ratio(feature_batch, feature_set):
    features = set()
    for fb in feature_batch:
        features = features | fb

    return len(features) / len(feature_set)


def feature_diversity(feature_batch):
    list_len = len(feature_batch)

    total_count = 0
    for i, x in enumerate(feature_batch):
        for j in range(i + 1, list_len):
            y = feature_batch[j]
            total_count += len(x & y)

    denominator = list_len * (list_len - 1) / 2
    return total_count / denominator


def mean_absolute_error(predicted, max_r, min_r, mae=True):
    total = 0
    for (r, p) in predicted:
        if p > max_r:
            p = max_r
        if p < min_r:
            p = min_r

        sub = p - r
        if mae:
            total += abs(sub)
        else:
            total += sub ** 2

    return total / len(predicted)


def root_mean_square_error(predicted, max_r, min_r):
    mse = mean_absolute_error(predicted, max_r, min_r, False)
    return math.sqrt(mse)

#构建实体字典
class EntityDictionary:
    #创建实体字典对象
    def __init__(self):
        self.idx2entity = [] #空列表，用于将索引映射到实体
        self.entity2idx = {} #空字典，用于将实体映射到索引

    #将实体添加到实体字典中的方法
    def add_entity(self, e):
        if e not in self.entity2idx: #如果实体 e 不在 entity2idx 字典中，表示该实体是一个新的实体
            self.entity2idx[e] = len(self.idx2entity) #将实体 e 添加到字典中，将其映射到当前 idx2entity 的长度（即下一个可用的索引）
            self.idx2entity.append(e)

    #获取实体字典中实体的数量的方法
    def __len__(self):
        return len(self.idx2entity) #返回列表的长度，即实体的数量

#加载和处理数据
class DataLoader:
    #创建数据加载器对象
    def __init__(self, data_path, index_dir, tokenizer, seq_len):
        self.user_dict = EntityDictionary() #存储用户和物品的实体字典
        self.item_dict = EntityDictionary()
        self.max_rating = float('-inf') #初始化 max_rating 为负无穷大，记录最大评分
        self.min_rating = float('inf')
        self.initialize(data_path) #调用 initialize 方法初始化数据
        self.feature_set = set() #初始化 feature_set 为空集合，用于存储特征集合
        self.tokenizer = tokenizer #存储传入的分词器 tokenizer
        self.seq_len = seq_len #序列长度 seq_len
        #调用 load_data 方法加载数据并将结果存储在 train、valid、test、user2feature 和 item2feature 中
        self.train, self.valid, self.test, self.user2feature, self.item2feature = self.load_data(data_path, index_dir)

    #初始化数据
    def initialize(self, data_path):
        assert os.path.exists(data_path) #断言数据路径 data_path 存在
        reviews = pickle.load(open(data_path, 'rb')) #用 pickle 加载数据文件 data_path，得到评论列表 reviews
        for review in reviews: #遍历每个评论 review
            self.user_dict.add_entity(review['user']) #将评论中的用户添加到 user_dict 实体字典中
            self.item_dict.add_entity(review['item']) #将评论中的物品添加到 item_dict 实体字典中
            rating = review['rating'] #更新最大评分和最小评分
            if self.max_rating < rating:
                self.max_rating = rating
            if self.min_rating > rating:
                self.min_rating = rating

    #加载数据的方法
    def load_data(self, data_path, index_dir):
        data = [] #初始化空列表 data，用于存储加载的数据
        reviews = pickle.load(open(data_path, 'rb')) #使用 pickle 加载数据文件 data_path，得到评论列表 reviews
        for review in reviews:
            (fea, adj, tem, sco) = review['template'] #从评论中提取模板、特征、得分等信息
            tokens = self.tokenizer(tem)['input_ids'] #使用分词器 tokenizer 对模板进行分词，并将其转换为文本形式
            text = self.tokenizer.decode(tokens[:self.seq_len])  # keep seq_len tokens at most
            #将评论的用户、物品、评分、文本和特征添加到 data 列表中
            data.append({'user': self.user_dict.entity2idx[review['user']],
                         'item': self.item_dict.entity2idx[review['item']],
                         'rating': review['rating'],
                         'text': text,
                         'feature': fea})
            #将特征添加到 feature_set 集合中
            self.feature_set.add(fea)

        train_index, valid_index, test_index = self.load_index(index_dir) #加载索引文件并获取训练集、验证集和测试集的索引
        train, valid, test = [], [], [] #根据索引从 data 列表中提取相应的样本,并将其存储在 train、valid、test 列表中。
        user2feature, item2feature = {}, {} #创建 user2feature 和 item2feature 字典，用于存储用户和物品对应的特征列表
        for idx in train_index: #遍历训练集索引，将相应的特征添加到 user2feature 和 item2feature 中
            review = data[idx]
            train.append(review) #获取评论中的用户、物品和特征分别赋值给变量 u、i 和 f
            u = review['user']
            i = review['item']
            f = review['feature']
            if u in user2feature: #用户 u 已经存在于 user2feature 字典中，则将特征 f 添加到对应的特征列表中
                user2feature[u].append(f)
            else: #否则，在 user2feature 字典中创建一个新的键值对，键为用户 u，值为包含特征 f 的列表
                user2feature[u] = [f]
            if i in item2feature:
                item2feature[i].append(f)
            else:
                item2feature[i] = [f]
        for idx in valid_index:
            valid.append(data[idx])
        for idx in test_index:
            test.append(data[idx])
        return train, valid, test, user2feature, item2feature

    #加载索引文件的方法
    def load_index(self, index_dir):
        assert os.path.exists(index_dir)
        with open(os.path.join(index_dir, 'train.index'), 'r') as f:
            train_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'validation.index'), 'r') as f:
            valid_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'test.index'), 'r') as f:
            test_index = [int(x) for x in f.readline().split(' ')]
        return train_index, valid_index, test_index

#将输入数据进行批处理
class Batchify:
    #接受data（输入数据列表）,tokenizer（用于对文本进行分词的分词器）,bos（表示文本序列的开头）,eos（表示文本序列的结尾）、batch_size（批处理大小，默认为128）和shuffle（是否对数据进行洗牌，默认为False）
    def __init__(self, data, tokenizer, bos, eos, batch_size=128, shuffle=False):
        #创建了空列表u、i、r、t和self.feature，用于存储用户、物品、评分、文本和特征信息。
        u, i, r, t, self.feature = [], [], [], [], []
        #遍历输入数据列表data,将每个数据项中的用户、物品、评分、文本和特征信息分别添加到对应的列表中
        #通过字符串格式化将文本序列的开头和结尾符号添加到文本中
        for x in data:
            u.append(x['user'])
            i.append(x['item'])
            r.append(x['rating'])
            t.append('{} {} {} '.format(bos, x['text'], eos))
            self.feature.append(x['feature'])

        #用tokenizer对文本序列t进行分词和编码处理
        #将文本序列t传递给它,设置padding=True进行填充,return_tensors='pt'表示返回PyTorch张量。返回的结果存储在encoded_inputs中
        encoded_inputs = tokenizer(t, padding=True, return_tensors='pt')
        #从encoded_inputs中获取编码后的序列input_ids和注意力掩码attention_mask,并将它们转换为连续张量（contiguous tensor
        self.seq = encoded_inputs['input_ids'].contiguous()
        self.mask = encoded_inputs['attention_mask'].contiguous()
        #将用户列表u、物品列表i、评分列表r分别转换为整数类型的张量，并同样转换为连续张量
        self.user = torch.tensor(u, dtype=torch.int64).contiguous()
        self.item = torch.tensor(i, dtype=torch.int64).contiguous()
        self.rating = torch.tensor(r, dtype=torch.float).contiguous()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    #用于获取下一个批次的数据
    def next_batch(self):
        #检查当前步数step是否等于总步数total_step
        if self.step == self.total_step:#相等，表示已经遍历完整个数据集，需要重置step为0
            self.step = 0
            if self.shuffle: #对索引列表index_list进行洗牌操作
                random.shuffle(self.index_list)

        #计算当前批次的起始索引start和结束索引offset
        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        #根据计算得到的索引范围start和offset，从用户、物品、评分、序列和掩码张量中获取对应批次的数据
        index = self.index_list[start:offset]
        user = self.user[index]  # (batch_size,)
        item = self.item[index]
        rating = self.rating[index]
        seq = self.seq[index]  # (batch_size, seq_len)
        mask = self.mask[index]
        return user, item, rating, seq, mask


#将输入数据进行批处理并生成用于模型输入的序列和特征
class Batchify2:
    def __init__(self, data, user2feature, item2feature, tokenizer, bos, eos, seq_len, batch_size=128, shuffle=False):
        #创建了三个空列表t、self.feature和features，用于存储文本序列、特征和处理后的特征信息
        t, self.feature, features = [], [], []
        #遍历输入数据列表data
        for x in data:
            #从user2feature字典中获取该用户对应的特征集合ufea
            ufea = set(user2feature[x['user']])
            ifea = set(item2feature[x['item']])
            #计算特征集合的交集intersection和并集差集difference
            intersection = ufea & ifea
            difference = ufea | ifea - intersection
            #将交集中的特征和差集中的特征转换为列表，使用空格连接起来，并将结果添加到features列表中
            features.append(' '.join(list(intersection) + list(difference)))
            #将文本序列按照指定格式进行拼接，添加到t列表中
            t.append('{} {} {}'.format(bos, x['text'], eos))
            #将输入数据项中的特征信息添加到self.feature列表中
            self.feature.append(x['feature'])

        #用tokenizer对文本序列t进行分词和编码处理
        encoded_inputs = tokenizer(t, padding=True, return_tensors='pt')
        self.seq = encoded_inputs['input_ids'].contiguous()
        self.mask = encoded_inputs['attention_mask'].contiguous()
        #将特征序列features传递给tokenizer，并进行填充和返回PyTorch张量的处理。返回的结果存储在encoded_features
        encoded_features = tokenizer(features, padding=True, return_tensors='pt')
        #从encoded_features中获取编码后的特征序列input_ids，并将其截断为指定长度seq_len，然后转换为连续张量
        self.prompt = encoded_features['input_ids'][:, :seq_len].contiguous()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data) #样本数量
        self.index_list = list(range(self.sample_num)) #索引列表
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    #获取下一个批次的数据
    def next_batch(self):
        if self.step == self.total_step: #已经遍历完所有数据
            self.step = 0
            if self.shuffle: #根据shuffle的值决定是否对索引列表index_list进行洗牌操作
                random.shuffle(self.index_list)

        start = self.step * self.batch_size #确定当前批次的数据范围
        offset = min(start + self.batch_size, self.sample_num) #确保不超出样本数量的范围
        self.step += 1
        index = self.index_list[start:offset]
        seq = self.seq[index]  # (batch_size, seq_len)
        mask = self.mask[index]
        prompt = self.prompt[index] #存储了编码后的特征序列
        return seq, mask, prompt

def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '


def postprocessing(string):
    '''
    adopted from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    '''
    string = re.sub('\'s', ' \'s', string)
    string = re.sub('\'m', ' \'m', string)
    string = re.sub('\'ve', ' \'ve', string)
    string = re.sub('n\'t', ' n\'t', string)
    string = re.sub('\'re', ' \'re', string)
    string = re.sub('\'d', ' \'d', string)
    string = re.sub('\'ll', ' \'ll', string)
    string = re.sub('\(', ' ( ', string)
    string = re.sub('\)', ' ) ', string)
    string = re.sub(',+', ' , ', string)
    string = re.sub(':+', ' , ', string)
    string = re.sub(';+', ' . ', string)
    string = re.sub('\.+', ' . ', string)
    string = re.sub('!+', ' ! ', string)
    string = re.sub('\?+', ' ? ', string)
    string = re.sub(' +', ' ', string).strip()
    return string

#将标记ID转换回文本形式
def ids2tokens(ids, tokenizer, eos):
    text = tokenizer.decode(ids) #使用分词器将标记ID解码为文本字符串。这一步将把标记ID序列转换成了可读的文本
    text = postprocessing(text)  # 处理标点符号，为标点符号添加空格: "good!" -> "good !"
    tokens = [] #创建一个空列表，用于存储最终的标记
    for token in text.split(): #遍历解码并经过后处理的文本中的每个标记
        if token == eos: #检查当前标记是否等于句子结尾的特殊标记eos
            break
        tokens.append(token) #将当前标记添加到tokens列表中
    return tokens #返回存储了解码、后处理并提取有效标记的列表

def criterion(y_pred, y_true, log_vars):
  loss = 0
  for i in range(len(y_pred)):
    precision = torch.exp(-log_vars[i]) #对对数方差进行指数运算得到精度
    diff = (y_pred[i]-y_true[i])**2.
    loss += torch.sum(precision * diff + log_vars[i], -1) #通过在最后一个维度上进行求和
  return torch.mean(loss)

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=2): #损失函数的数量，默认为 2
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True) #创建一个包含 num 个元素的张量，初始值为 1 。requires_grad=True 表示这些参数需要计算梯度。
        self.params = torch.nn.Parameter(params) #将 params 张量包装成 nn.Parameter 对象，使其成为模型的可学习参数。

    def forward(self, *x):
        loss_sum = 0 #初始化一个变量 loss_sum，用于累计损失值
        for i, loss in enumerate(x): #迭代输入的损失值。使用 enumerate 函数可以同时获取损失值和对应的索引。
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)#根据损失值和权重参数计算加权损失并累加到 loss_sum 中
        return loss_sum , self.params

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, param_groups=None, rho=0.5, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        if param_groups is None:
            param_groups = [{'params': params, 'rho': rho, 'adaptive': adaptive}]

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for group in self.param_groups:
            rho = group['rho']  
            adaptive = group['adaptive']
            grad_norm = self._grad_norm(group['params'])
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if adaptive else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self, params):
        shared_device = params[0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group, p in zip(self.param_groups, params)
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)