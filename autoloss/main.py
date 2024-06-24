from torch import optim
from myloss import UncertaintyWeightedLoss
from module import Mymodel

model = Mymodel()

uwl = UncertaintyWeightedLoss(2)	# have 2 losses
loss_1 = t_loss      # first loss
loss_2 = r_loss      # second loss

# learnable parameters
optimizer = optim.Adam([
                {'params': model.parameters()},
                {'params': uwl.parameters(), 'weight_decay': 0}
            ])
""" optimizer = optim.SGD([
                {'params': model.parameters(), 'lr': args.lr },
                {'params': awl.parameters(), 'weight_decay': 0}
            ])  """

for i in range(epoch):
    for data, label1, label2 in data_loader:
        # forward
        pred1, pred2 = Mymoduel(data)	
        # calculate losses
        loss1 = loss_1(pred1, label1)
        loss2 = loss_2(pred2, label2)
        # weigh losses
        loss_sum = awl(loss1, loss2)
        # backward
        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()