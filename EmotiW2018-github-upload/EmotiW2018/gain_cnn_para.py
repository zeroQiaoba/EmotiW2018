#coding:utf8
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# Hyper Parameters
num_epochs = 1000
batch_size = 1
learning_rate = 0.001

class CNN(nn.Module):
    def __init__(self, modelNum):
        super(CNN, self).__init__() # [1, 10, 7120, 3]
        self.conv1x1 = nn.Conv2d(modelNum, 1, kernel_size=1, padding=0)
        
    def forward(self, x):
        out = self.conv1x1(x)# [1, 10, 7120, 3] -> [1, 1, 7120, 3]
        out = out.view(-1, 7)
        return out #[7120, 3]



# Target: Train the Model, and gain paras with best top3
# Input:  modelNum, int
#         X, list # [5, 4846, 3] ->[modelNum, sample, class]
#         y, list # # [7120] ->[sample]
# Output: best_top3, float
#         best_paras, dict
def gain_cnn_para(modelNum, X, y):        
    
    best_top1 = 0
    best_paras = ""

    cnn = CNN(modelNum)
    X = torch.FloatTensor(np.array(X)).unsqueeze(0)# [1, 10, 7120, 3]
    y = torch.LongTensor(np.array(y).astype(int)).squeeze()# [7120]
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Train the Model
    for epoch in range(num_epochs):
        optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
        images = Variable(X) #[1, 6, 1156, 7]
        labels = Variable(y) #[bacth]
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images) #[7120, 3]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        top1 = testModel(cnn, X, y)
        
         # 保存top1最大的模型参数
        bias = copy.copy(cnn.conv1x1.bias.data[0])
        weight = copy.copy(cnn.conv1x1.weight.data.squeeze().numpy())
        #print ('bias=', bias, '   weight=', weight)
        #print ('top1=%.2f' % (top1*100))
        if top1 > best_top1:
            best_top1 = top1
            best_paras = {'bias': bias,'weight':weight}
            
    return best_top1, best_paras


# Target: Test the Model, and gain top1 and top3 of the cnn
# Input:  cnn, model
#         X, torch.tensor
#         y, torch.tensor
# Output: None
def testModel(cnn, X, y):
    cnn.eval()
    
    images = Variable(X)
    labels = Variable(y) #[7120]
    probs = cnn(images) #[7120, 3]

    ## calculate top1
    probs = probs.data.numpy()
    trueLabels = labels.data.numpy()
    predLabels = np.argmax(probs, axis = 1) # [sample, 3]
    top1 = np.sum(trueLabels == predLabels)/float(len(predLabels))

    cnn.train()

    return top1

   
    
# Target: Gain fusion result
# Input: para with {'bias': bias,'weight':weight}, dict
#        para['bias'], array
#        para['weight'], array
#        probs, list [nModel, sample, 3]
# Output: fusion result, array [sample, 3]
def gain_cnn_fusion(para, probs):
    assert len(para['weight']) == len(probs)
    result = para['bias']
    for i in range(len(para['weight'])):
        result += para['weight'][i]*probs[i]
    return result

    