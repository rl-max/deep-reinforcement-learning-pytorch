import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from torch.distributions import Normal
from collections import deque

#Hyperparameters
EPISODES = 1000
discount_factor = 0.98
log_alpha = torch.tensor(np.log(0.01), requires_grad=True)
target_entropy = -1.0
train_interval = 10
q_lr = 0.0005
policy_lr = 0.0002
alpha_lr = 0.001
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
        #print(x.shape, a.shape)
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc(x))
        return self.q(x)

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mu = nn.Linear(128, 1)
        self.sigma = nn.Linear(128, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        dist = Normal(self.mu(x), F.softplus(self.sigma(x)))
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        action = 2*torch.tanh(action)
        return action, log_prob

def make_batch(buffer):
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

def train(networks, buffer, optimizers):
    obs, acts, rewards, next_obs, done = make_batch(buffer)
    q1, q1_target, q2, q2_target, pi = networks
    q1_optimizer, q2_optimizer, pi_optimizer, alpha_optimizer = optimizers
    
    next_acts, log_prob = pi(next_obs)
    q_target = torch.min(q1_target(next_obs, next_acts), q2_target(next_obs, next_acts))
    target = rewards.view(-1, 1) + discount_factor * done.view(-1, 1) * \
                                   (q_target - torch.exp(log_alpha)*log_prob)
    
    q1_loss = F.smooth_l1_loss(q1(obs, acts.view(-1, 1)), target.detach())
    q2_loss = F.smooth_l1_loss(q2(obs, acts.view(-1, 1)), target.detach())
    
    q1_optimizer.zero_grad(); q1_loss.backward(); q1_optimizer.step()
    q2_optimizer.zero_grad(); q2_loss.backward(); q2_optimizer.step()
    
    sampled_a, log_prob = pi(obs)
    q_value = torch.min(q1(obs, sampled_a), q2(obs, sampled_a))
    policy_obj = -q_value + torch.exp(log_alpha)*log_prob
    pi_optimizer.zero_grad()
    policy_obj.mean().backward()
    pi_optimizer.step()
    
    alpha_obj = -torch.exp(log_alpha)*(log_prob.detach() + target_entropy)
    alpha_optimizer.zero_grad()
    alpha_obj.mean().backward()
    alpha_optimizer.step()
    
def soft_update(t_net, net, target_ratio):
    for t_param, param in zip(t_net.parameters(), net.parameters()):
        t_param.data.copy_(t_param.data*target_ratio + param.data*(1-target_ratio))

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    q1net, q1_target, q2net, q2_target, pinet = QNet(), QNet(), QNet(), \
                                                QNet(), PolicyNet()
    q1_target.load_state_dict(q1net.state_dict())
    q2_target.load_state_dict(q2net.state_dict())
    q1_optimizer = optim.Adam(q1net.parameters(), lr=q_lr)
    q2_optimizer = optim.Adam(q2net.parameters(), lr=q_lr)
    pi_optimizer = optim.Adam(pinet.parameters(), lr=policy_lr)
    alpha_optimizer = optim.Adam([log_alpha], lr=alpha_lr)
    
    buffer = deque(maxlen=buffer_size)
    score, step =  0.0, 0
    for ep in range(EPISODES):
        done = False
        obs = env.reset()
        while not done:
            action, _ = pinet(torch.tensor(obs).float())
            next_obs, reward, done, info = env.step([action.item()])
            buffer.append((obs, action.item(), reward/10.0, next_obs, done))
            score += reward
            step += 1
            obs = next_obs
            
            if step%train_interval==0 and len(buffer) > start_train:
                train((q1net, q1_target, q2net, q2_target, pinet), buffer, \
                      (q1_optimizer, q2_optimizer, pi_optimizer, alpha_optimizer))
                soft_update(q1_target, q1net, target_update)
                soft_update(q2_target, q2net, target_update)
                    
        if ep%10==0 and ep!=0:
            print('episode:{}, buffer_size:{}, alpha:{}, avg_score:{}'.format(
                  ep, len(buffer), torch.exp(log_alpha).item(), score/10.0))
            score = 0.0
    env.close()