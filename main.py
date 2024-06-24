import os
import math
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, AdamW
from module import RecReg
from utils import AutomaticWeightedLoss, rouge_score, bleu_score, DataLoader, Batchify, now_time, ids2tokens, unique_sentence_percent, \
    root_mean_square_error, mean_absolute_error, feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity


parser = argparse.ArgumentParser(description='PErsonalized Prompt Learning for Explainable Recommendation (PEPLER)')
parser.add_argument('--data_path', type=str, default=None,
                    help='path for loading the pickle data')
parser.add_argument('--index_dir', type=str, default=None,
                    help='load indexes')
parser.add_argument('--model_path', type=str, default=None,
                    help='load model')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--checkpoint', type=str, default='./best',
                    help='directory to save the final model')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--endure_times', type=int, default=5,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--use_mf', action='store_true',
                    help='otherwise MLP')
parser.add_argument('--words', type=int, default=20,
                    help='number of words to generate for each sample')
args = parser.parse_args()

if args.data_path is None:
    parser.error('--data_path should be provided for loading data')
if args.index_dir is None:
    parser.error('--index_dir should be provided for loading data splits')

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

if torch.cuda.is_available():
    if not args.cuda:
        print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
device = torch.device('cuda' if args.cuda else 'cpu')

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)
    model_path = os.path.join(args.checkpoint, 'model.pt')
prediction_path = os.path.join(args.checkpoint, args.outf)

###############################################################################
# Load data
###############################################################################

print(now_time() + 'Loading data')
bos = '<bos>' #beginning of sequence
eos = '<eos>' #end of sequence
pad = '<pad>' #padding
tokenizer = GPT2Tokenizer.from_pretrained(r'/root/autodl-tmp/poetry/llm/gpt2', local_files_only=True, bos_token=bos, eos_token=eos, pad_token=pad)
corpus = DataLoader(args.data_path, args.index_dir, tokenizer, args.words)
feature_set = corpus.feature_set
train_data = Batchify(corpus.train, tokenizer, bos, eos, args.batch_size, shuffle=True)
val_data = Batchify(corpus.valid, tokenizer, bos, eos, args.batch_size)
test_data = Batchify(corpus.test, tokenizer, bos, eos, args.batch_size)

###############################################################################
# Build the model
###############################################################################

nuser = len(corpus.user_dict)
nitem = len(corpus.item_dict)
ntoken = len(tokenizer) #计算了分词器中标记的数量，即标记的个数
model = RecReg.from_pretrained(r'/root/autodl-tmp/poetry/llm/gpt2', nuser, nitem, use_mf=args.use_mf, local_files_only=True)
model.resize_token_embeddings(ntoken)  # 调整模型的嵌入层大小以匹配标记的词序列
model = torch.load(args.model_path)#load model  Specify the device on which the parameters are loaded
model.to(device)
awl = AutomaticWeightedLoss(2)
rating_criterion = nn.MSELoss() #定义了均方误差损失函数，用于评估推荐任务的输出与目标之间的差异

""" optimizer = optim.AdamW([
                {'params': model.parameters() , 'lr': args.lr},
                {'params': awl.parameters(),'lr': args.lr, 'weight_decay': 0}
            ]) """

optimizer = optim.Adam([
                {'params': model.parameters(), 'lr': args.lr},
                {'params': awl.parameters(), 'weight_decay': 0}
            ])
            
""" optimizer = optim.SGD([
                {'params': model.parameters() , 'lr': 0.001},
                {'params': awl.parameters(),'lr': 0.001, 'weight_decay': 0}
            ])
 """

###############################################################################
# Training code
###############################################################################
#uncertainty w


def train(data):
    # Turn on training mode which enables dropout.
    model.train()
    text_loss = 0. #记录文本损失的累计值
    rating_loss = 0. #记录推荐损失的累计值
    total_sample = 0 #记录总样本数量
    while True:
        user, item, rating, seq, mask = data.next_batch()  # data.step += 1
        user = user.to(device)  # (batch_size,)
        item = item.to(device)
        rating = rating.to(device)
        seq = seq.to(device)  # (batch_size, seq_len)
        mask = mask.to(device)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad() #将优化器的梯度置零,清除之前的梯度信息
        outputs, rating_p = model(user, item, seq, mask) #通过模型进行前向传播，得到输出 outputs 和评分预测 rating_p
        
        t_loss = outputs.loss #计算文本损失 t_loss 
        r_loss = rating_criterion(rating_p, rating) #计算评分损失 r_loss
        #loss = args.text_reg * t_loss + args.rating_reg * r_loss  #计算总损失 loss，其中包括文本损失和评分损失，并根据权重参数进行加权

        loss , autow = awl(t_loss, r_loss)

        loss.backward()
        optimizer.step()

        batch_size = user.size(0)
        #更新文本损失和评分损失的累计值以及样本数量
        text_loss += batch_size * t_loss.item()
        rating_loss += batch_size * r_loss.item()
        total_sample += batch_size

        if data.step % args.log_interval == 0 or data.step == data.total_step:
            cur_t_loss = text_loss / total_sample
            cur_r_loss = rating_loss / total_sample
            print(now_time() + 'text ppl {:4.4f} | rating loss {:4.4f} | {:5d}/{:5d} batches'.format(
                math.exp(cur_t_loss), cur_r_loss, data.step, data.total_step))
            text_loss = 0.
            rating_loss = 0.
            total_sample = 0
        if data.step == data.total_step:
            break
    return autow


def evaluate(data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    text_loss = 0. #初始化变量，记录文本损失、评分损失和样本数量
    rating_loss = 0.
    total_sample = 0
    with torch.no_grad():
        while True:
            user, item, rating, seq, mask = data.next_batch()  # data.step += 1
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            rating = rating.to(device)
            seq = seq.to(device)  # (batch_size, seq_len)
            mask = mask.to(device)
            outputs, rating_p = model(user, item, seq, mask)
            t_loss = outputs.loss #计算文本损失
            r_loss = rating_criterion(rating_p, rating) #计算评分损失

            batch_size = user.size(0)
            #更新文本损失和评分损失的累计值以及样本数量
            text_loss += batch_size * t_loss.item()
            rating_loss += batch_size * r_loss.item()
            total_sample += batch_size

            if data.step == data.total_step:
                break
    return text_loss / total_sample, rating_loss / total_sample


def generate(data):
    # Turn on evaluation mode which disables dropout.贪心算法，概率最大的单词作为每一步的预测
    model.eval()
    #初始化空列表 idss_predict 和 rating_predict，用于存储生成的文本序列和评分预测
    idss_predict = []
    rating_predict = []
    with torch.no_grad():
        while True:
            user, item, rating, seq, _ = data.next_batch()  # data.step += 1
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            text = seq[:, :1].to(device)  #  # 初始化 text，将其设为序列的第一个词（bos）：seq[:, :1], (batch_size, 1)
            for idx in range(seq.size(1)):
                # produce a word at each step
                if idx == 0:  #如果 idx 是第一个位置，通过模型进行前向传播，得到输出 outputs 和评分预测 rating_p
                    outputs, rating_p = model(user, item, text, None)
                    rating_predict.extend(rating_p.tolist()) #将评分预测结果添加到 rating_predict 列表中
                else:
                    outputs, _ = model(user, item, text, None, False) #对于其他位置，通过模型进行前向传播
                last_token = outputs.logits[:, -1, :]  # 获取最后一个词的概率分布 last_token, (batch_size, ntoken)
                word_prob = torch.softmax(last_token, dim=-1)
                token = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1), pick the one with the largest probability
                text = torch.cat([text, token], 1)  # (batch_size, len++)
            ids = text[:, 1:].tolist()  #当生成完成后，将生成的文本序列转换为列表形式，并将其添加到, (batch_size, seq_len)
            idss_predict.extend(ids)

            if data.step == data.total_step:
                break
    return idss_predict, rating_predict

# Loop over epochs.
best_val_loss = float('inf') #记录最佳验证损失
endure_count = 0
for epoch in range(1, args.epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    autow = train(train_data)
    val_t_loss, val_r_loss = evaluate(val_data) #获取验证集的文本损失 val_t_loss 和评分损失 val_r_loss
    val_loss = val_t_loss + val_r_loss #计算验证集的总损失 val_loss
    
    #打印验证集的文本困惑度、评分损失和总损失
    print(now_time() + 'text ppl {:4.4f} | rating loss {:4.4f} | valid loss {:4.4f} on validation'.format(
        math.exp(val_t_loss), val_r_loss, val_loss))
    
    #weight
    for i in range(len(autow)):
        print(f"weight[{i}]: {autow[i]}") 
    
    # Save the model if the validation loss is the best we've seen so far.
    #如果当前的验证损失是目前为止最佳的，将其作为新的最佳验证损失，并将模型保存到文件中
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        with open(args.model_path, 'wb') as f:
            torch.save(model, f)
    else:
        endure_count += 1
        print(now_time() + 'Endured {} time(s)'.format(endure_count))
        if endure_count == args.endure_times:
            print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
            break

# Load the best saved model.
with open(args.model_path, 'rb') as f:
    model = torch.load(f).to(device)

# Run on test data.
test_t_loss, test_r_loss = evaluate(test_data)
print('=' * 89)
print(now_time() + 'text ppl {:4.4f} | rating loss {:4.4f} on test | End of training'.format(math.exp(test_t_loss), test_r_loss))
print(now_time() + 'Generating text')
idss_predicted, rating_predicted = generate(test_data)    
# rating
predicted_rating = [(r, p) for (r, p) in zip(test_data.rating.tolist(), rating_predicted)]
RMSE = root_mean_square_error(predicted_rating, corpus.max_rating, corpus.min_rating)
print(now_time() + 'RMSE {:7.4f}'.format(RMSE))
MAE = mean_absolute_error(predicted_rating, corpus.max_rating, corpus.min_rating)
print(now_time() + 'MAE {:7.4f}'.format(MAE))
# text
tokens_test = [ids2tokens(ids[1:], tokenizer, eos) for ids in test_data.seq.tolist()]
tokens_predict = [ids2tokens(ids, tokenizer, eos) for ids in idss_predicted]
BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
print(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
print(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))
USR, USN = unique_sentence_percent(tokens_predict)
print(now_time() + 'USR {:7.4f} | USN {:7}'.format(USR, USN))
feature_batch = feature_detect(tokens_predict, feature_set)
DIV = feature_diversity(feature_batch)  # time-consuming
print(now_time() + 'DIV {:7.4f}'.format(DIV))
FCR = feature_coverage_ratio(feature_batch, feature_set)
print(now_time() + 'FCR {:7.4f}'.format(FCR))
FMR = feature_matching_ratio(feature_batch, test_data.feature)
print(now_time() + 'FMR {:7.4f}'.format(FMR))
text_test = [' '.join(tokens) for tokens in tokens_test]
text_predict = [' '.join(tokens) for tokens in tokens_predict]
ROUGE = rouge_score(text_test, text_predict)  # a dictionary
for (k, v) in ROUGE.items():
    print(now_time() + '{} {:7.4f}'.format(k, v))
text_out = ''
for (real, fake) in zip(text_test, text_predict):
    text_out += '{}\n{}\n\n'.format(real, fake)
with open(prediction_path, 'w', encoding='utf-8') as f:
    f.write(text_out)
print(now_time() + 'Generated text saved to ({})'.format(prediction_path))