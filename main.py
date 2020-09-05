import torch.nn as nn
from  TreeBankDataSet import CustomDataset
from torch.autograd import Variable
import torch
from torch.utils.data import random_split,DataLoader
from tqdm import tqdm
from pos_tagger import LSTM
import matplotlib.pyplot as plt


def plot(all_acc):
    plt.figure()
    plt.title('Accuracy/Epoch')
    plt.plot(all_acc)
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy value')
    plt.show()
    
treebank_dataset = CustomDataset(dname='treebank')    
vocab_size = treebank_dataset.get_vocab_size()
target_size = treebank_dataset.get_target_size()

EMBEDDING_DIM = 300
hidden_dim = 400
output_dim = target_size
learning_rate = 1e-4
num_epochs =10
batch_size=64

train_size = int(0.8 * len(treebank_dataset))
test_size = len(treebank_dataset) - train_size
train_dataset, test_dataset = random_split(treebank_dataset, [train_size, test_size])


train_loader = DataLoader(train_dataset, batch_size,shuffle=True,drop_last=True)
test_loader = DataLoader(test_dataset, batch_size,shuffle=True,drop_last=True)


    
model = LSTM(EMBEDDING_DIM, hidden_dim, output_dim,vocab_size,batch_size)

if torch.cuda.is_available():
    model.cuda()
     
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_list = []
all_acc=[]
for epoch in range(num_epochs):
    training_loss = []
    for batch_no,(data, labels) in  tqdm(enumerate(train_loader),total=len(train_loader),desc='Epoch {}'.format(epoch+1)):
        if torch.cuda.is_available():
            data = Variable(data.cuda())
            labels = Variable(labels.cuda())
        else:
            data = Variable(data)
            labels = Variable(labels)
          
        optimizer.zero_grad()
        
        outputs = model(data)
        
        
        outputs=outputs.view(-1,target_size)
        labels=labels.view(-1)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        training_loss.append(loss.item())
    print('training_loss_per_batch')
    print(sum(training_loss)/len(training_loss))
        
    accuracy=0
    correct = 0
    total = 0
   
    
    for batch_no,(test_data, test_labels) in tqdm(enumerate(test_loader),total=len(test_loader),desc='Epoch {}'.format(epoch+1)):
        
        if torch.cuda.is_available():
            test_data = Variable(test_data.cuda())
            test_labels = Variable(test_labels.cuda())
        else:
            data = Variable(test_data)
            labels = Variable(test_labels)
        
        test_outputs = model(test_data)
        _, predicted = torch.max(test_outputs.data, 2)
        predicted = predicted.view(-1)
        test_labels = test_labels.view(-1)
        
        total += len(test_labels)
        correct += (predicted == test_labels).sum()
        accuracy = 100 * correct / float(total)
        
    print('Epoch: {}. Loss: {}. Accuracy: {}'.format(epoch+1, loss.item(), accuracy))
    abc=accuracy.cpu()
    c=abc.numpy()
    all_acc.append(c)
plot(all_acc)
    