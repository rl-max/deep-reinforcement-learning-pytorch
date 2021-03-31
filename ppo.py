import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#Hyperparameters
EPISODES = 10000
learning_rate = 0.0002
discount_factor, lmbda = 0.98, 0.5
train_interval = 20
train_iter = 3
epsilon = 0.1

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
    #mini-batch
    obs, acts, probs, rewards, next_obs, done = [], [], [], [], [], []
    for transition in samples:
        s, a, p, r, s_, d = transition
        d = 0.0 if d else 1.0
        obs.append(s); acts.append(a); probs.append(p); rewards.append(r)
        next_obs.append(s_), done.append(d)
    
    obs, acts, probs, rewards, next_obs, done = torch.tensor(obs).float(), \
    torch.tensor(acts), torch.tensor(probs).float(), torch.tensor(rewards).float(),\
    torch.tensor(next_obs).float(), torch.tensor(done)
    
    #train
    for _ in range(train_iter):
        target = rewards.view(-1, 1) + discount_factor * net.v(next_obs) * done.view(-1, 1)
        td = target - net.v(obs)
        #Implementation of GAE(Generalized Advantage Estimation)
        advantage, R = [], 0.0
        for delta in torch.flip(td, dims=[0, 1]):
            R = delta + discount_factor * lmbda * R
            advantage.append(R)
        advantage.reverse()
        advantage = torch.tensor(advantage).float().unsqueeze(1)
        
        pi_a = net.pi(obs, softmax_dim=1).gather(1, acts.view(-1, 1))
        probs = probs.view(-1, 1)
        ratio = torch.exp(torch.log(pi_a) - torch.log(probs).detach()) 
        clipped = torch.clamp(ratio, 1-epsilon, 1+epsilon)
        
        p_loss = -torch.min(ratio*advantage, clipped*advantage)
        v_loss = F.smooth_l1_loss(net.v(obs), target.detach())
        loss = (p_loss + v_loss).mean()
        
        optimizer.zero_grad()
        loss.backward()
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
            samples.append((obs, action, prob[action].item(), reward/100.0, next_obs, done))
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