import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque

#Hyperparameters
EPISODES = 10000
learning_rate = 0.0005
discount_fact = 0.98
buffer_size, start_train = 50000, 2000
batch_size = 32

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

def minibatch_and_train(net, target_net, optimizer, buffer):
    mini_batch = random.sample(buffer, batch_size)
    
    obs, acts, rewards, next_obs, done = [], [], [], [], []
    for sample in mini_batch:
        s, a, r, s_, d = sample
        d = 0.0 if d else 1.0
        obs.append(s); acts.append(a); rewards.append(r); 
        next_obs.append(s_); done.append(d)
    
    obs, acts, rewards, next_obs, done = torch.tensor(obs).float(),\
    torch.tensor(acts), torch.tensor(rewards).float(), torch.tensor(next_obs).float(),\
    torch.tensor(done)
    
    target_q = rewards + discount_fact * done * target_net(next_obs).max(dim=1)[0]
    target_q = target_q.view(-1, 1)
    q = net(obs).gather(1, acts.view(-1, 1))
    loss = F.smooth_l1_loss(q, target_q.detach())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    net, target_net = QNet(), QNet()
    target_net.load_state_dict(net.state_dict())
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    buffer = deque(maxlen=buffer_size)
    score, step = 0, 0
    epsilon, epsilon_decay = 0.6, 1-1e-5
    target_interval = 20
    
    for ep in range(EPISODES):
        obs = env.reset()
        done = False
        while not done:
            q_value = net(torch.tensor(obs).float())
            rand = random.random()
            if rand < epsilon:
                action = random.randint(0, 1)
            else:
                action = q_value.argmax().item()
            next_obs, reward, done, info = env.step(action)
            buffer.append((obs, action, reward/100.0, next_obs, done))
            obs = next_obs
            step += 1
            score += reward
            epsilon *= epsilon_decay
            
        if len(buffer) > start_train:
            minibatch_and_train(net, target_net, optimizer, buffer)
        
        if ep%target_interval==0 and ep!=0:
            target_net.load_state_dict(net.state_dict())
            
        if ep%10==0 and ep!=0:
            print('episode:{}, step:{}, avg_score:{}, len_buffer:{}, epsilon:{}'.format(ep, step, \
                  score/10.0, len(buffer), epsilon))
            score = 0
    env.close()