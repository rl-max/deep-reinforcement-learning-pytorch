import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

#Note: This is not trained well. Use this for reference only.

EPISODES = 10000
learning_rate = 0.0005
discount_fact = 0.98
buffer_size, start_train = 100000, 2000
batch_size = 32
min_sprt, max_sprt = -2, 2
num_sprt = 51
interval = (max_sprt - min_sprt)/(num_sprt - 1)

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, 256)
        self.act1 = nn.Linear(256, num_sprt)
        self.act2 = nn.Linear(256, num_sprt)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        act1 = F.softmax(self.act1(x), dim=1)
        act2 = F.softmax(self.act2(x), dim=1)
        return act1, act2

def mini_batch(buffer):
    batch = random.sample(buffer, batch_size)
    obs, acts, rewards, next_obs, done = [], [], [], [], []
    for sample in batch:
        s, a, r, s_, d = sample
        d = 0.0 if d else 1.0
        obs.append(s); acts.append(a); rewards.append(r); 
        next_obs.append(s_); done.append(d)

    return torch.tensor(obs).float(), torch.tensor(acts), \
           torch.tensor(rewards).float(), torch.tensor(next_obs).float(), \
           torch.tensor(done)

def train(net, target_net, optimizer, buffer):
    obs, acts, rewards, next_obs, done = mini_batch(buffer)
    q1_dist_, q2_dist_ = target_net(next_obs)
    
    sprt = torch.arange(min_sprt, max_sprt+1e-2, interval).unsqueeze(0)
    support = sprt.repeat(batch_size, 1)
    target_sprt = rewards.view(-1, 1) + discount_fact * done.view(-1, 1) * support
    target_sprt = torch.clamp(target_sprt, min_sprt, max_sprt)
    dists_ = []
    for idx, (q1, q2) in enumerate(zip(expect(q1_dist_), expect(q2_dist_))):
        if q1 >= q2: dists_.append(q1_dist_[idx])
        else:        dists_.append(q2_dist_[idx])
    dists_ = torch.stack(dists_, dim=0).float().detach()
    #projection
    target_dists = []
    for sprt, t_sprt, dist in zip(support, target_sprt, dists_):
        t_dists = []
        for idx, ts in enumerate(t_sprt):
            sub = np.abs(sprt - ts)
            sub[sub > interval] = interval
            #proportion between two probablities
            proportion = (interval-sub)/interval
            td = np.array(dist[idx] * proportion)
            t_dists.append(td)
        t_dist = np.sum(np.array(t_dists), axis=0)
        target_dists.append(t_dist)
    target_dists = torch.tensor(target_dists).float()
    
    dists = []
    q1_dist, q2_dist = net(obs)
    for q1, q2, act in zip(q1_dist, q2_dist, acts):
        if act.item() == 0:  dists.append(q1)
        else:                dists.append(q2)
    dists = torch.stack(dists, dim=0).float()
    
    loss = F.kl_div(dists, target_dists.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
      
def expect(dists): #function calculating expectation value
    q = []
    for dist in dists:
        support = torch.arange(min_sprt, max_sprt+1e-2, interval)
        q_val = torch.sum(support * dist)
        q.append(q_val)
    return torch.tensor(q).float()

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    net, target_net = QNet(), QNet()
    target_net.load_state_dict(net.state_dict())
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    buffer = deque(maxlen=buffer_size)
    score, step = 0.0, 0
    epsilon, epsilon_decay = 0.4, 1-5e-5
    target_interval = 5
    
    for ep in range(EPISODES):
        obs = env.reset()
        done = False
        while not done:
            q1_dist, q2_dist = net(torch.tensor(obs).unsqueeze(0).float())
            q1_val, q2_val = expect(q1_dist).item(), expect(q2_dist).item()
            rand = random.random()
            if rand < epsilon:
                action = random.randint(0, 1)
            else:
                action = 0 if q1_val >= q2_val else 1
            next_obs, reward, done, info = env.step(action)
            buffer.append((obs, action, reward/10.0, next_obs, done))
            obs = next_obs
            step += 1
            score += reward
            epsilon *= epsilon_decay
            
        if len(buffer) > start_train:
            train(net, target_net, optimizer, buffer)
            
        if ep%target_interval==0 and ep!=0:
            target_net.load_state_dict(net.state_dict())
            
        if ep%10==0 and ep!=0:
            print('episode:{}, step:{}, avg_score:{}, len_buffer:{}, epsilon:{}'.format(ep, step, \
                  score/10.0, len(buffer), epsilon))
            score = 0
    env.close()