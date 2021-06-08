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
log_alpha = torch.tensor(np.log(0.1), requires_grad=True)
target_entropy = -4.0
train_interval = 10
q_lr = 0.0005
policy_lr = 0.0002
discrim_lr = 0.0005
alpha_lr = 0.001
buffer_size, start_train = 100000, 2000
batch_size = 32
target_update = 0.995

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_s = nn.Linear(24, 256)
        self.fc_z = nn.Linear(8, 256)
        self.hidden = nn.Linear(512, 512)
        self.mu = nn.Linear(512, 4)
        self.sigma = nn.Linear(512, 4)
    
    def forward(self, obs, z):
        obs = F.relu(self.fc_s(obs))
        z = F.relu(self.fc_z(z))
        x = torch.cat([obs, z], dim=1)
        x = F.relu(self.hidden(x))
        mu, sigma = self.mu(x), F.softplus(self.sigma(x))
        dists = Normal(mu, sigma)
        actions = dists.rsample()
        log_probs = torch.sum(dists.log_prob(actions), dim=1, keepdim=True)
        return actions, log_probs
    
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_s = nn.Linear(24, 256)
        self.fc_z = nn.Linear(8, 256)
        self.fc_a = nn.Linear(4, 256)
        self.fc = nn.Linear(768, 768)
        self.q = nn.Linear(768, 1)
    
    def forward(self, obs, z, a):
        obs = F.relu(self.fc_s(obs))
        z = F.relu(self.fc_z(z))
        a = F.relu(self.fc_a(a))
        x = torch.cat([obs, z, a], dim=1)
        x = F.relu(self.fc(x))
        return self.q(x)
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(24, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.discrim = nn.Linear(512, 16)
    
    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        log_prob = F.log_softmax(self.discrim(x), dim=0)
        return log_prob

def make_batch(buffer):
    mini_batch = random.sample(buffer, batch_size)
    obs, skills, acts, rewards, next_obs, done = [], [], [], [], [], []
    for samples in mini_batch:
        s, z, a, r, s_, d = samples
        d = 0.0 if d else 1.0
        obs.append(s); skills.append(z); acts.append(a); 
        rewards.append(r); next_obs.append(s_); done.append(d)
    
    return torch.tensor(obs).float(), torch.tensor(skills).float(), \
           torch.tensor(acts).float(), torch.tensor(rewards).float(), \
           torch.tensor(next_obs).float(), torch.tensor(done)

def train(networks, buffer, optimizers):
    obs, skills, acts, rewards, next_obs, done = make_batch(buffer)
    
    q1, q1_target, q2, q2_target, pi = networks
    q1_optimizer, q2_optimizer, pi_optimizer, alpha_optimizer = optimizers

    next_acts, log_prob = pi(next_obs, skills)
    q_target = torch.min(q1_target(next_obs, skills, next_acts), \
                         q2_target(next_obs, skills, next_acts))
    target = rewards.view(-1, 1) + discount_factor * done.view(-1, 1) * \
                                   (q_target - torch.exp(log_alpha)*log_prob)
    
    q1_loss = F.smooth_l1_loss(q1(obs, skills, acts), target.detach())
    q2_loss = F.smooth_l1_loss(q2(obs, skills, acts), target.detach())
    
    q1_optimizer.zero_grad(); q1_loss.backward(); q1_optimizer.step()
    q2_optimizer.zero_grad(); q2_loss.backward(); q2_optimizer.step()
    
    sampled_a, log_prob = pi(obs, skills)
    q_value = torch.min(q1(obs, skills, sampled_a), q2(obs, skills, sampled_a))
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
    env = gym.make('BipedalWalkerHardcore-v3')
    q1net, q1_target, q2net, q2_target, pinet, discriminator = QNet(),\
    QNet(), QNet(), QNet(), PolicyNet(), Discriminator()
    q1_target.load_state_dict(q1net.state_dict())
    q2_target.load_state_dict(q2net.state_dict())
    q1_optimizer = optim.Adam(q1net.parameters(), lr=q_lr)
    q2_optimizer = optim.Adam(q2net.parameters(), lr=q_lr)
    pi_optimizer = optim.Adam(pinet.parameters(), lr=policy_lr)
    alpha_optimizer = optim.Adam([log_alpha], lr=alpha_lr)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=discrim_lr)
    
    buffer = deque(maxlen=buffer_size)
    score, step =  0.0, 0
    for ep in range(EPISODES):
        done = False
        obs = env.reset()
        skills = [np.random.rand(8) for _ in range(16)]
        select_z  = random.randint(0, 15)
        z = skills[select_z]
        while not done:
            action, _ = pinet(torch.tensor(obs).unsqueeze(0).float(),\
                              torch.tensor(z).unsqueeze(0).float())
            env.render()
            next_obs, _, done, info = env.step(action[0].detach())
            reward = discriminator(torch.tensor(next_obs).float())[select_z]
            buffer.append((obs, z, action[0].detach().numpy(), reward.item()/100.0, next_obs, done))
            score += reward.item()
            step += 1
            obs = next_obs
            
            #discriminator update
            disc_optimizer.zero_grad()
            (-reward).backward()
            disc_optimizer.step()
            
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