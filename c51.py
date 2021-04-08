import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

#Note: This is not being trained well. Use this for reference only.

EPISODES = 10000
learning_rate = 0.0005
discount_factor = 0.9
buffer_size, start_train = 100000, 2000
batch_size = 32
min_sprt, max_sprt = 0, 2
num_sprt = 51
interval = (max_sprt - min_sprt)/(num_sprt - 1)
num_act = 2

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, 256)
        self.acts = [nn.Linear(256, num_sprt) for _ in range(num_act)]
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        acts = [F.softmax(self.acts[idx](x), dim=1) for idx in range(num_act)]
        return acts

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
    next_actdists = target_net(next_obs)
    support = torch.arange(min_sprt, max_sprt+1e-2, interval).unsqueeze(0)
    supports = support.repeat(batch_size, 1)
    target_sprts = rewards.view(-1, 1) + discount_factor * done.view(-1, 1) * supports
    target_sprts = torch.clamp(target_sprts, min_sprt, max_sprt)
    next_actvals = torch.stack([expect(next_actdists[idx]) for idx in range(num_act)], dim=0)
    next_maxact = torch.argmax(next_actvals, dim=0)
    next_dists = []
    for idx, maxact in enumerate(next_maxact):
        next_dists.append(next_actdists[maxact][idx])
    next_dists = torch.stack(next_dists, dim=0).detach()
    #projection
    target_dists = []
    for supprt, target_sprt, dist in zip(supports, target_sprts, next_dists):
        sub_dists = []
        for idx, ts in enumerate(target_sprt):
            diff = np.abs(supprt - ts)
            diff[diff > interval] = interval
            proportion = (interval - diff)/interval
            t_d = np.array(dist[idx] * proportion)
            sub_dists.append(t_d)
        t_dist = np.sum(np.array(sub_dists), axis=0)
        target_dists.append(t_dist)
    target_dists = torch.tensor(target_dists).float()
    dists = []
    actdists = net(obs)
    for idx, act in enumerate(acts):
        dists.append(actdists[act][idx])
    dists = torch.stack(dists, dim=0).float()
    
    loss = F.kl_div(dists, target_dists.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
def expect(dists): #function calculating expectation value
    support = torch.arange(min_sprt, max_sprt+1e-2, interval)
    supports = support.repeat(len(dists), 1)
    q_val = torch.sum(supports * dists, dim=1)
    return q_val

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    net, target_net = QNet(), QNet()
    target_net.load_state_dict(net.state_dict())
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    buffer = deque(maxlen=buffer_size)
    score, step = 0.0, 0
    epsilon, epsilon_decay = 0.4, 1-5e-5
    target_interval = 10
    
    for ep in range(EPISODES):
        obs = env.reset()
        done = False
        while not done:
            acts_dist = net(torch.tensor(obs).unsqueeze(0).float())
            acts_val = np.array([expect(acts_dist[idx]).item() for idx in range(num_act)])
            rand = random.random()
            if rand < epsilon:
                action = random.randint(0, num_act-1)
            else:
                action = acts_val.argmax()
            next_obs, reward, done, info = env.step(action)
            buffer.append((obs, action, reward/50.0, next_obs, done))
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