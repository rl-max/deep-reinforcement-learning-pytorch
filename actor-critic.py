import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#Hyperparameters
EPISODES = 10000
learning_rate = 0.0002
discount_factor = 0.98
train_interval = 20

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.p = nn.Linear(128, 2)
        self.value = nn.Linear(128, 1)
    
    def pi(self, x, softmax_dim):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        prob = F.softmax(self.p(x), dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.value(x)

def train(net, optimizer, samples):
    obs, acts, rewards, next_obs, done = [], [], [], [], []
    for transition in samples:
        s, a, r, s_, d = transition
        d = 0.0 if d else 1.0
        obs.append(s); acts.append(a); rewards.append(r)
        next_obs.append(s_), done.append(d)
    
    obs, acts, rewards, next_obs, done = torch.tensor(obs).float(),\
    torch.tensor(acts), torch.tensor(rewards).float(), torch.tensor(next_obs).float(),\
    torch.tensor(done)
    
    target = rewards.view(-1, 1) + discount_factor * net.v(next_obs) * done.view(-1, 1)
    td = target - net.v(obs)
    prob = net.pi(obs, softmax_dim=1).gather(1, acts.view(-1, 1))
    
    loss = -torch.log(prob) * td.detach() + F.smooth_l1_loss(net.v(obs), target.detach())
    
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    net = Network()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    samples, score, step = [], 0.0, 0
    
    for ep in range(EPISODES):
        obs = env.reset()
        done = False
        while not done:
            prob = net.pi(torch.tensor(obs).float(), softmax_dim=0)
            prob_ = Categorical(prob)
            action = prob_.sample().item()
            next_obs, reward, done, info = env.step(action)
            samples.append((obs, action, reward/100.0, next_obs, done))
            score += reward
            step += 1
            obs = next_obs
            
            if step%train_interval==0:
                train(net, optimizer, samples)
                samples = []
        
        if ep%10==0 and ep!=0:
            print('episode:{}, num_train:{}, avg_score:{}'.format(ep, \
                   step//train_interval, score/10.0))
            score = 0.0
    env.close()