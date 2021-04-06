import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import deque

#Hyperparameters
EPISODES = 10000
learning_rate = 0.0005
discount_factor = 0.98
buffer_size, start_train = 50000, 2000
epsilon, alpha, beta = 0.1, 0.6, 0.4 #for PER
batch_size = 64

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.v = nn.Linear(128, 1)
        self.adv = nn.Linear(128, 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.v(x)
        adv = self.adv(x)
        mean_adv = 0.5*torch.sum(adv, dim=1, keepdim=True)
        q = v + adv - mean_adv
        return q

def mini_batch(buffer, priority):
    real_p = priority[priority!=None] #get real(calculated from TD-error) priority
    max_p = max(real_p) if len(real_p)!=0 else 1.0
    #priority of unvisited data should be max-priority
    prior = np.array([p**alpha if p!=None else max_p**alpha for p in priority])
    prob = prior/sum(prior)
    
    indices = np.random.choice(len(buffer), batch_size, p=prob)
    mini_batch = np.array(buffer, dtype=object)[indices]
    indices_prob = prob[indices]
    
    obs, acts, rewards, next_obs, done = [], [], [], [], []
    for sample in mini_batch:
        s, a, r, s_, d = sample
        d = 0.0 if d else 1.0
        obs.append(s); acts.append(a); rewards.append(r); 
        next_obs.append(s_); done.append(d)
    
    return torch.tensor(obs).float(), torch.tensor(acts), torch.tensor(rewards).float(), \
           torch.tensor(next_obs).float(), torch.tensor(done), indices, \
           torch.tensor(indices_prob).float()
    
def train(net, target_net, optimizer, buffer, priority):
    priority = np.array(priority)
    obs, acts, rewards, next_obs, done, indices, prob = mini_batch(buffer, priority)
    
    target_a = net(next_obs).argmax(dim=1).view(-1, 1)
    q_target = target_net(next_obs).gather(1, target_a)
    target_q = rewards.view(-1, 1) + discount_factor * done.view(-1, 1) * q_target
    q = net(obs).gather(1, acts.view(-1, 1))
    
    weight = (len(buffer)*prob) ** -beta #Importance-sampling weight from PER
    loss = weight.view(-1, 1) * F.smooth_l1_loss(q, target_q.detach(), reduce=False)
    
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()
    
    #update priority
    prior = (torch.abs(target_q - q) + epsilon).view(-1)
    prior = prior.detach().numpy()
    priority[indices] = prior
    priority = deque(priority, maxlen=buffer_size)
    return priority
    
if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    net, target_net = QNet(), QNet()
    target_net.load_state_dict(net.state_dict())
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    buffer = deque(maxlen=buffer_size)
    priority = deque(maxlen=buffer_size)
    score, step = 0, 0
    epsilon, epsilon_decay = 0.6, 1-1e-5
    target_interval = 20
    
    for ep in range(EPISODES):
        obs = env.reset()
        done = False
        while not done:
            q_value = net(torch.tensor(obs).unsqueeze(0).float())
            rand = random.random()
            if rand < epsilon:
                action = random.randint(0, 1)
            else:
                action = q_value.argmax().item()
            next_obs, reward, done, info = env.step(action)
            #Priority is initialized by 'None'
            buffer.append((obs, action, reward/100.0, next_obs, done))
            priority.append(None)
            obs = next_obs
            step += 1
            score += reward
            epsilon *= epsilon_decay
            
        if len(buffer) > start_train:
            priority = train(net, target_net, optimizer, buffer, priority)
        
        if ep%target_interval==0 and ep!=0:
            target_net.load_state_dict(net.state_dict())
            
        if ep%10==0 and ep!=0:
            print('episode:{}, step:{}, avg_score:{}, len_buffer:{}, epsilon:{}'.format(ep, step, \
                  score/10.0, len(buffer), epsilon))
            score = 0.0
    env.close()