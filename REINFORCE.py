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

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=0)
        return x

def train(net, optimizer, samples):
    R, loss = 0, 0
    optimizer.zero_grad()
    for prob, r in reversed(samples):
        R = r + discount_factor * R
        loss = -torch.log(prob) * R
        loss.backward()
    optimizer.step()
    
if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    net = PolicyNet()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    score = 0.0
    
    for ep in range(EPISODES):
        obs = env.reset()
        samples = []
        done = False
        while not done:
            prob = net(torch.tensor(obs).float())
            prob_ = Categorical(prob)
            action = prob_.sample().item()
            next_obs, reward, done, info = env.step(action)
            samples.append((prob[action], reward/100.0))
            score += reward
            obs = next_obs

        train(net, optimizer, samples)
        
        if ep%10==0 and ep!=0:
            print('episode:{}, avg_score:{}'.format(ep, score/10.0))
            score = 0.0
    env.close()