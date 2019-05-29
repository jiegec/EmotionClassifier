import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

import numpy as np

import sys
import re

data = []
num_emotions = 8

spaces = re.compile("\s")

skip_words = set(["a", "href", "nbsp", "href", "http:", "news", "sina", "com", "cn", "target", "blank"])
vocab_set = set()

if torch.cuda.is_available():
    device = torch.device("cuda:%s" % sys.argv[3])
else:
    device = torch.device("cpu")

max_vote_count = [0] * num_emotions

# read train file
file = open(sys.argv[1], "r")
max_words = 4096
for line in file:
    parts = spaces.split(line)

    max_vote = 0
    votes = []
    vote_sum = 0.0
    for i in range(num_emotions):
        num = int(parts[i + 2][3:])
        votes.append(num)
        vote_sum += num
    max_vote = np.argmax(votes)
    max_vote_count[max_vote] += 1.0
    words = []
    for i in range(2 + num_emotions, len(parts)):
        if parts[i] not in skip_words and len(parts[i]) > 0:
            words.append(parts[i])
            vocab_set.add(parts[i])
    data.append((max_vote, words))


vocab = list(sorted(vocab_set))
vocab_map = dict()
vocab_size = len(vocab)

for i in range(0, vocab_size):
    vocab_map[vocab[i]] = i

# read test file
test_file = open(sys.argv[2], "r")
test_data = []
for line in test_file:
    parts = spaces.split(line)

    max_vote = 0
    votes = []
    for i in range(num_emotions):
        num = int(parts[i + 2][3:])
        votes.append(num)
    max_vote = np.argmax(votes)
    words = [0] * max_words
    words_len = 0
    for i in range(2 + num_emotions, len(parts)):
        if parts[i] in vocab_set:
            words[words_len] = vocab_map[parts[i]]
            words_len += 1
    test_data.append((torch.tensor([words], dtype=torch.long).to(device), torch.tensor(max_vote, dtype=torch.long)))

batch_size = 256
batch_input_data = []
input_data = []
for i in range(0, len(data), batch_size):
    local_inputs = []
    local_labels = []
    for (votes, words) in data[i:(i+batch_size)]:
        inputs = [0] * max_words
        for i in range(len(words)):
            inputs[i] = vocab_map[words[i]]
        local_inputs.append([inputs])
        local_labels.append(votes)

        input_data.append((torch.tensor([inputs], dtype=torch.long).to(device), torch.tensor([votes], dtype=torch.long).to(device)))
    batch_input_data.append((torch.tensor(local_inputs, dtype=torch.long).to(device), torch.tensor(local_labels, dtype=torch.long).to(device)))


print('got', len(data), 'news and', len(vocab), 'words')

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 256).to(device)
        self.conv1 = nn.Conv1d(256, 64, 20).to(device)
        self.conv2 = nn.Conv1d(256, 64, 10).to(device)
        self.conv3 = nn.Conv1d(256, 64, 5).to(device)
        self.max1 = nn.MaxPool1d(32).to(device)
        self.max2 = nn.MaxPool1d(32).to(device)
        self.max3 = nn.MaxPool1d(32).to(device)
        self.dropout = nn.Dropout(0.2).to(device)
        self.linear = nn.Linear(32 * 762, num_emotions).to(device)

        nn.init.xavier_uniform(self.embedding.weight)
        nn.init.xavier_uniform(self.conv1.weight)
        nn.init.xavier_uniform(self.conv2.weight)
        nn.init.xavier_uniform(self.conv3.weight)
        nn.init.xavier_uniform(self.linear.weight)

    def forward(self, x):
        x = self.embedding(x)
        #print(x.size())
        x = x.permute(0, 2, 1)
        #print(x.size())
        x1 = F.relu(self.conv1(x))
        #print(x.size())
        x1 = F.relu(self.max1(x1))
        #print(x1.size())
        x2 = F.relu(self.conv2(x))
        #print(x.size())
        x2 = F.relu(self.max2(x2))
        #print(x2.size())
        x3 = F.relu(self.conv3(x))
        #print(x.size())
        x3 = F.relu(self.max3(x3))
        #print(x3.size())
        x = torch.cat([x1, x2, x3])
        x = F.relu(self.dropout(x))
        #print(x.size())
        x = x.view(1, -1)
        #print(x.size())
        x = self.linear(x)[0]
        return x

model = CNN()
criterion = nn.CrossEntropyLoss().to(device)
#criterion = nn.CrossEntropyLoss(weight=torch.tensor(max_vote_count, dtype=torch.float)).to(device)
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def print_accuracy():
    # on training set
    correct = 0
    wrong = 0
    input_dist = [0] * num_emotions
    output_dist = [0] * num_emotions
    for data in input_data:
        inputs, labels = data
        outputs = model(inputs)

        input_ans = labels
        output_ans = np.argmax(list(outputs))

        input_dist[input_ans] += 1
        output_dist[output_ans] += 1

        #print('expected %d, got %d %s' % (input_ans, output_ans, outputs.tolist()))
        if input_ans == output_ans:
            correct += 1
        else:
            wrong += 1

    print('train correct/all: %d/%d, dist: %s %s' % (correct, correct + wrong, input_dist, output_dist))

    # on test set
    correct = 0
    wrong = 0
    for data in test_data:
        inputs, labels = data
        outputs = model(inputs)

        input_ans = labels
        output_ans = np.argmax(list(outputs))

        #print('expected %d, got %d %s' % (input_ans, output_ans, outputs.tolist()))
        if input_ans == output_ans:
            correct += 1
        else:
            wrong += 1

    print('test correct/all: %d/%d' % (correct, correct + wrong))


print("before training")
print_accuracy()
for epoch in range(1000):
    total_loss = 0.0
    for data in batch_input_data:
        inputs, labels = data

        optimizer.zero_grad()
        outputs = torch.stack([model(inp) for inp in inputs])
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print('epoch %d loss %.3f' % (epoch, total_loss))
    print_accuracy()