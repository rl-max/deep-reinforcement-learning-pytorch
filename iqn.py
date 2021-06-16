import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque

#Hyperparameters
EPISODES = 10000
learning_rate = 0.0001
discount_factor = 0.98
buffer_size, start_train = 100000, 2000
batch_size = 32
target_sprt = 64
pred_sprt = 32
embed_dim = 2
cvar_eta = 0.75
k = 1.0
#constant
state_space, action_space = 8, 4
PI = 3.1416

class Quantile(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(embed_dim, 256)
        self.fc1 = nn.Linear(state_space, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.acts = nn.Linear(256, action_space)
    
    def forward(self, obs, tau):
        taus = tau.view(-1, 1).expand(-1, embed_dim)
        embed_tau = taus * torch.arange(0, embed_dim) * PI
        embed_tau = F.relu(self.embed(torch.cos(embed_tau)))
        obs = F.relu(self.fc1(obs))
        x = obs * embed_tau
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.acts(x)

def mini_batch(buffer):
    mini_batch = random.sample(buffer, batch_size)
    obs, acts, rewards, next_obs, done = [], [], [], [], []
    
    for sample in mini_batch:
        s, a, r, s_, d = sample
        d = 0.0 if d else 1.0
        obs.append(s); acts.append(a); rewards.append(r); 
        next_obs.append(s_); done.append(d)
    
    return torch.tensor(obs).float(), torch.tensor(acts), \
           torch.tensor(rewards).float(), torch.tensor(next_obs).float(),\
           torch.tensor(done)

def train(net, target_net, optimizer, buffer):
    obs, acts, rewards, next_obs, done = mini_batch(buffer)
    
    next_q = [predict(target_net, next_obs, cvar_eta)[0] for _ in range(target_sprt)]
    next_q = torch.stack(next_q, dim=2)
    max_act = next_q.mean(dim=2).argmax(dim=1)
    next_qval = [next_q[idx][max_a] for idx, max_a in enumerate(max_act)] 
    next_qval = torch.stack(next_qval, dim=0)
    target_q = rewards.view(-1, 1) + discount_factor * next_qval
    
    current_q, probs = [], []
    for _ in range(pred_sprt):
        val, taus = predict(net, obs, cvar_eta)
        current_q.append(val); probs.append(taus)
    current_q = torch.stack(current_q, dim=2)
    curr_qval = [current_q[idx][a] for idx, a in enumerate(acts)]
    curr_qval = torch.stack(curr_qval, dim=0)
    probs = torch.stack(probs, dim=1).unsqueeze(1)
    
    #Quantile Regresion Loss
    target_q = target_q.view(batch_size, -1, 1).expand(-1, target_sprt, pred_sprt).detach()
    curr_q = curr_qval.view(batch_size, 1, -1).expand(-1, target_sprt, pred_sprt)
    diff = target_q - curr_q
    soft_diff = torch.where(torch.abs(diff)<=k, 0.5*torch.pow(diff, 2), \
                            k*(torch.abs(diff) - 0.5*k))
    s_diff1 = soft_diff * probs
    s_diff2 = soft_diff * (1 - probs)
    error = torch.where(diff>=0, s_diff1, s_diff2)
    loss = torch.sum(error) / (batch_size * target_sprt)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def predict(net, obs, cvar_eta):
    tau_ = cvar_eta * torch.rand(obs.size(0))
    return net(obs, tau_), tau_

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    net, target_net = Quantile(), Quantile()
    target_net.load_state_dict(net.state_dict())
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, eps=0.01/32)
    
    buffer = deque(maxlen=buffer_size)
    score, step = 0, 0
    epsilon, epsilon_decay = 0.4, 1-5e-6
    target_interval = 15
    
    for ep in range(EPISODES):
        obs = env.reset()
        done = False
        while not done:
            qvals = [predict(net, torch.tensor(obs).unsqueeze(0).float(), cvar_eta)[0] \
                     for _ in range(pred_sprt)]
                
            qvals = torch.stack(qvals, dim=0).mean(dim=0)
            rand = random.random()
            if rand < epsilon:
                action = random.randint(0, action_space-1)
            else:
                action = qvals.argmax().item()
            next_obs, reward, done, info = env.step(action)
            buffer.append((obs, action, reward/100.0, next_obs, done))
            obs = next_obs
            step += 1
            score += reward
            epsilon *= epsilon_decay
            
        if len(buffer) > start_train:
            train(net, target_net, optimizer, buffer)
        
        if ep%target_interval==0 and ep!=0:
            target_net.load_state_dict(net.state_dict())
            
        if ep%10==0 and ep!=0:
            print('episode:{}, step:{}, avg_score:{}, len_buffer:{}, epsilon:{}'.format(ep,\
                 step, score/10.0, len(buffer), epsilon))
            score = 0
    env.close()