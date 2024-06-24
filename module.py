from transformers import GPT2LMHeadModel
import torch.nn as nn
import torch
import copy

class UIPrompt:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, nuser, nitem, freezeLM=True, **kwargs):#从预训练模型中加载模型，并根据给定的参数nuser和nitem初始化用户和项目的嵌入层。
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # freeze pretrained model parameters
        if freezeLM:
            for param in model.parameters():
                param.requires_grad = False

        model.init_prompt(nuser, nitem)
        return model

    def init_prompt(self, nuser, nitem):#初始化用户和项目的嵌入层
        self.src_len = 2#表示源长度为2
        emsize = self.transformer.wte.weight.size(1)  # 根据模型的嵌入层的维度大小（这里假设为768）768
        self.user_embeddings = nn.Embedding(nuser, emsize)#定义了用户和项目的嵌入层
        self.item_embeddings = nn.Embedding(nitem, emsize)

        initrange = 0.1#使用均匀分布随机初始化嵌入层的权重。
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, user, item, text, mask, ignore_index=-100):
        #获取输入user、item、text和mask的相关信息，包括设备类型和批次大小。
        device = user.device
        batch_size = user.size(0)

        # embeddings
        u_src = self.user_embeddings(user)  # (batch_size, emsize)
        i_src = self.item_embeddings(item)  # (batch_size, emsize)
        w_src = self.transformer.wte(text)  # (batch_size, tgt_len, emsize)
        src = torch.cat([u_src.unsqueeze(1), i_src.unsqueeze(1), w_src], 1)  # (batch_size, total_len, emsize)
        #通过用户和项目的嵌入层和文本的嵌入层，得到输入的嵌入表示src
        #将用户和项目的嵌入通过unsqueeze操作添加一个维度，然后将它们与文本的嵌入拼接在一起，得到形状为(batch_size, total_len, emsize)的src

        if mask is None:
            # auto-regressive generation
            return super().forward(inputs_embeds=src)
        else:
            # training
            # input padding
            #生成用于训练的输入pad_input，通过将mask在左侧添加一个全为1的填充向量
            pad_left = torch.ones((batch_size, self.src_len), dtype=torch.int64).to(device)
            pad_input = torch.cat([pad_left, mask], 1)  # (batch_size, total_len)

            # prediction for training
            #通过比较mask中元素是否为1，将text中的<pad>标记替换为ignore_index，得到pred_right
            pred_left = torch.full((batch_size, self.src_len), ignore_index, dtype=torch.int64).to(device)  # (batch_size, src_len)
            pred_right = torch.where(mask == 1, text, torch.tensor(ignore_index).to(device))  # replace <pad> with ignore_index
            #将pred_left和pred_right拼接在一起，得到用于训练的预测值prediction
            prediction = torch.cat([pred_left, pred_right], 1)  # (batch_size, total_len)
            
            #调用父类的forward方法，同时传入pad_input、src和prediction，进行训练
            return super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction)


class ContinuousPromptLearning(UIPrompt, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)


class FeaturePrompt:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def forward(self, context, explanation, exp_mask, ignore_index=-100):
        device = context.device
        #将context和explanation在维度1上拼接，得到形状为(batch_size, total_len)的text
        text = torch.cat([context, explanation], 1)  # (batch_size, total_len)
        #文本的嵌入层self.transformer.wte将text转换为嵌入表示src，形状为(batch_size, total_len, emsize)
        src = self.transformer.wte(text)  # (batch_size, total_len, emsize)

        if exp_mask is None:
            # auto-regressive generation自回归生成
            return super().forward(inputs_embeds=src)
        else:
            # training
            # input padding首先生成用于训练的输入pad_input，通过将exp_mask在左侧添加一个全为1的填充向量
            pad_left = torch.ones_like(context, dtype=torch.int64).to(device)
            pad_input = torch.cat([pad_left, exp_mask], 1)  # (batch_size, total_len)

            # prediction for training比较exp_mask中元素是否为1，将explanation中的<pad>标记替换为ignore_index，得到pred_right
            pred_left = torch.full_like(context, ignore_index, dtype=torch.int64).to(device)  # (batch_size, src_len)
            pred_right = torch.where(exp_mask == 1, explanation, torch.tensor(ignore_index).to(device))  # replace <pad> with ignore_index
            #将pred_left和pred_right拼接在一起，得到用于训练的预测值prediction
            prediction = torch.cat([pred_left, pred_right], 1)  # (batch_size, total_len)

            #传入pad_input作为注意力掩码（attention_mask），src作为输入嵌入（inputs_embeds），prediction作为标签（labels），进行训练
            return super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction)


class DiscretePromptLearning(FeaturePrompt, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)

#定义MF的神经网络模型类
class MF(nn.Module):
    def __init__(self):
        super(MF, self).__init__()

    def forward(self, user, item):  # (batch_size, emsize)
        #通过对user和item进行逐元素相乘,然后对每个样本的乘积结果进行求和（sum）得到一个评分值（rating）,return评分结果
        rating = torch.sum(user * item, 1)  # (batch_size,)
        return rating


def _get_clones(module, N):
    #用于复制一个模型的多个副本，N表示要创建的副本数量
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

#定义一个多层感知机（MLP）模型类MLP
class MLP(nn.Module):
    def __init__(self, emsize, hidden_size=400, num_layers=2):
        super(MLP, self).__init__()
        self.first_layer = nn.Linear(emsize * 2, hidden_size)#输入，输出维度
        self.last_layer = nn.Linear(hidden_size, 1)
        layer = nn.Linear(hidden_size, hidden_size)
        self.layers = _get_clones(layer, num_layers)#复制了layer层,复制的数量由num_layers指定,可创建多个相同的隐藏层,用于构建多层感知机
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):#初始化模型的权重。遍历模型的所有线性层，用均匀分布的随机数进行权重初始化，偏置初始化为0
        initrange = 0.1
        self.first_layer.weight.data.uniform_(-initrange, initrange)
        self.first_layer.bias.data.zero_()
        self.last_layer.weight.data.uniform_(-initrange, initrange)
        self.last_layer.bias.data.zero_()
        for layer in self.layers:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.zero_()

    def forward(self, user, item):  # (batch_size, emsize)
        ui_cat = torch.cat([user, item], 1)  # 在维度1上进行拼接 (batch_size, emsize * 2)
        hidden = self.sigmoid(self.first_layer(ui_cat))  # (batch_size, hidden_size)
        for layer in self.layers:
            hidden = self.sigmoid(layer(hidden))  # (batch_size, hidden_size)
        rating = torch.squeeze(self.last_layer(hidden))  # (batch_size,)
        return rating

class IntegratedModel(nn.Module):
    def __init__(self, mf_model, mlp_model, user, item):
        super(IntegratedModel, self).__init__()
        self.mf_model = mf_model()
        self.mlp_model = mlp_model()

    def forward(self, user, item):
        mf_rating = self.mf_model()
        mlp_rating = self.mlp_model() 
        integrated_rating = (mf_rating + mlp_rating) / 2  # 可根据需求定义集成策略
        return integrated_rating


#带有正则化的用户-物品推荐模型
class UIPromptWithReg:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, nuser, nitem, use_mf=True, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        model.init_prompt(nuser, nitem, use_mf)
        return model

    def init_prompt(self, nuser, nitem, use_mf):#nuser（用户嵌入矩阵的大小）、nitem（物品嵌入矩阵的大小
        self.src_len = 2 #输入序列的长度
        emsize = self.transformer.wte.weight.size(1)  # 嵌入层的权重矩阵 默认768
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
        if use_mf:
            self.rec = MF()
        else:
            self.rec = MLP(emsize)

        #对self.user_embeddings和self.item_embeddings的权重进行均匀分布的随机初始化
        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, user, item, text, mask, rating_prediction=True, ignore_index=-100):
        device = user.device #device被设置为user的设备
        batch_size = user.size(0) #batch_size被设置为user的第一个维度大小

        # embeddings
        u_src = self.user_embeddings(user)  # (batch_size, emsize)
        i_src = self.item_embeddings(item)  # (batch_size, emsize)
        w_src = self.transformer.wte(text)  # (batch_size, tgt_len, emsize)
        #通过在维度1上拼接u_src、i_src和w_src，得到src，其中total_len是输入序列的总长度
        src = torch.cat([u_src.unsqueeze(1), i_src.unsqueeze(1), w_src], 1)  # (batch_size, total_len, emsize)

        if rating_prediction:
            rating = self.rec(u_src, i_src)  # (batch_size,)
        else:
            rating = None

        if mask is None:
            # auto-regressive generation表示进行自回归生成，直接调用
            return super().forward(inputs_embeds=src), rating
        else:
            # training表示进行训练，需要进行填充和预测：
            # input padding
            pad_left = torch.ones((batch_size, self.src_len), dtype=torch.int64).to(device)
            pad_input = torch.cat([pad_left, mask], 1)  # (batch_size, total_len)

            # prediction for training
            pred_left = torch.full((batch_size, self.src_len), ignore_index, dtype=torch.int64).to(device)  # (batch_size, src_len)
            pred_right = torch.where(mask == 1, text, torch.tensor(ignore_index).to(device))  # replace <pad> with ignore_index
            prediction = torch.cat([pred_left, pred_right], 1)  # (batch_size, total_len)

            return super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction), rating
            
class RecReg(UIPromptWithReg, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)

