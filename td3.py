import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque

#Hyperparameters
EPISODES = 1000
q_lr = 0.0005
mu_lr = 0.0001
discount_factor = 0.98
train_interval = 10
policy_delay = 20
buffer_size, start_train = 100000, 2000
batch_size = 32
target_update = 0.995

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.obs = nn.Linear(3, 128)
        self.act = nn.Linear(1, 128)
        self.fc = nn.Linear(256, 256)
        self.q = nn.Linear(256, 1)
    
    def forward(self, x, a):
        x = F.relu(self.obs(x))
        a = F.relu(self.act(a))
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc(x))
        return self.q(x)

class ActionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.action = nn.Linear(128, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        a = torch.tanh(self.action(x))
        return 2*a

def make_bach(buffer):
    mini_batch = random.sample(buffer, batch_size)
    obs, acts, rewards, next_obs, done = [], [], [], [], []
    for samples in mini_batch:
        s, a, r, s_, d = samples
        d = 0.0 if d else 1.0
        obs.append(s); acts.append(a); rewards.append(r); 
        next_obs.append(s_); done.append(d)
    
    return torch.tensor(obs).float(), torch.tensor(acts).float(), \
           torch.tensor(rewards).float(), torch.tensor(next_obs).float(), \
           torch.tensor(done)
           
def train(networks, buffer, optimizers, step):
    obs, acts, rewards, next_obs, done = make_bach(buffer)
    q1, q1_target, q2, q2_target, mu, mu_target = networks
    q1_optimizer, q2_optimizer, mu_optimizer = optimizers
    
    noise = torch.clamp(torch.randn(batch_size, 1)*0.5, -0.2, 0.2)
    target_act = torch.clamp(mu_target(next_obs) + noise, -2.0, 2.0)
    q_target = torch.min(q1_target(next_obs, target_act),\
                         q2_target(next_obs, target_act))
    target = rewards.view(-1, 1) + discount_factor * done.view(-1, 1) * q_target
                                     
    q1_loss = F.smooth_l1_loss(q1(obs, acts.view(-1, 1)), target.detach())
    q2_loss = F.smooth_l1_loss(q2(obs, acts.view(-1, 1)), target.detach())
    q1_optimizer.zero_grad(); q1_loss.backward(); q1_optimizer.step()
    q2_optimizer.zero_grad(); q2_loss.backward(); q2_optimizer.step()
    
    if step%policy_delay==0:
        mu_obj = -q1(obs, mu(obs)).mean()
        mu_optimizer.zero_grad()
        mu_obj.backward()
        mu_optimizer.step()

#Implementation of soft-update
def soft_update(t_net, net, target_ratio):
    for t_param, param in zip(t_net.parameters(), net.parameters()):
        t_param.data.copy_(t_param.data*target_ratio + param.data*(1-target_ratio))

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    q1net, q1_target, q2net, q2_target, munet, mu_target = QNet(), QNet(), QNet(), QNet(),\
                                                     ActionNet(), ActionNet()
    q1_target.load_state_dict(q1net.state_dict())
    q2_target.load_state_dict(q2net.state_dict())
    mu_target.load_state_dict(munet.state_dict())
    q1_optimizer = optim.Adam(q1net.parameters(), lr=q_lr)
    q2_optimizer = optim.Adam(q2net.parameters(), lr=q_lr)
    mu_optimizer = optim.Adam(munet.parameters(), lr=mu_lr)
    
    buffer = deque(maxlen=buffer_size)
    score, step = 0.0, 0
    
    for ep in range(EPISODES):
        done = False
        obs = env.reset()
        while not done:
            a = munet(torch.tensor(obs).float())
            noise = torch.randn(1) * 0.5
            action = torch.clamp(a+noise, -2.0, 2.0).item()
            next_obs, reward, done, info = env.step([action])
            buffer.append((obs, action, reward/100.0, next_obs, done))
            score += reward
            step += 1
            obs = next_obs
            
            if step%train_interval==0 and len(buffer) > start_train:
                train((q1net, q1_target, q2net, q2_target, munet, mu_target), buffer, \
                      (q1_optimizer, q2_optimizer, mu_optimizer), step)
                soft_update(q1_target, q1net, target_update)
                soft_update(q2_target, q2net, target_update)
                soft_update(mu_target, munet, target_update)
                
        if ep%10==0 and ep!=0:
            print('epsidoes:{}, buffer_size:{}, avg_score:{}'.format(ep, len(buffer), score/10.0))
            score = 0.0