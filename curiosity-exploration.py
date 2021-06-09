import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque

#Hyperparameters
EPISODES = 10000
discount_fact = 0.98
buffer_size, start_train = 50000, 100
batch_size = 64
reward_eta = 1.0
q_lr = 0.0005
encoder_lr = 0.0005
forward_lr, inverse_lr = 0.0005, 0.0005

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 8, stride=3)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.batch_norm = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 5, stride=2)
        self.fc1 = nn.Linear(4800, 512)
        self.q = nn.Linear(512, 4)
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.batch_norm(x)
        x = F.leaky_relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        return self.q(x)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 7, stride=3)
        self.batch_norm = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=2)
        self.encode = nn.Conv2d(32, 1, 1)
    
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.batch_norm(x)
        x = F.leaky_relu(self.conv2(x))
        encode = F.tanh(self.encode(x))
        return encode

class Forward_model(nn.Module):
    def __init__(self):
        #action shape: (1, 1)
        #state shape: (1, 1, 32, 24)
        super().__init__()
        self.conv_s = nn.Conv2d(1, 1, 1)
        self.conv_a = nn.Conv2d(1, 1, 1)
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 32, 2, stride=2)
        self.trans = nn.ConvTranspose2d(32, 1, 6, stride=2)
    
    def forward(self, x, a):
        a = a.view(a.size(0), 1, 1, 1).repeat(1, 1, 32, 24)
        a = self.conv_a(a)
        x = F.leaky_relu(self.conv_s(x))
        x = torch.add(x, a)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        next_x = torch.tanh(self.trans(x))
        return next_x

class Inverse_model(nn.Module):
    def __init__(self):
        super().__init__()
        #state shape: (1, 1, 32, 24)
        self.encode1 = nn.Conv2d(1, 1, 3)
        self.encode2 = nn.Conv2d(1, 1, 3)
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 32, 2, stride=2)
        self.fc = nn.Linear(3744, 4)
    
    def forward(self, pre_x, curr_x):
        pre_x = self.encode1(pre_x)
        curr_x = self.encode2(curr_x)
        x = F.leaky_relu(torch.add(pre_x, curr_x))
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def mini_batch(buffer):
    mini_batch = random.sample(buffer, batch_size)
    obs, acts, rewards, next_obs, done = [], [], [], [], []
    for sample in mini_batch:
        s, a, r, s_, d = sample
        d = 0.0 if d else 1.0
        obs.append(s); acts.append(a); rewards.append(r); 
        next_obs.append(s_); done.append(d)
    
    return torch.tensor(obs).float(), torch.tensor(acts), torch.tensor(rewards).float(),\
    torch.tensor(next_obs).float(), torch.tensor(done)

def train(networks, optimizers, buffer):
    q_net, q_target, encode, forward, inverse = networks
    q_optim, en_optim, fwd_optim, inv_optim = optimizers
    obs, acts, rewards, next_obs, done = mini_batch(buffer)
    
    acts_pred = inverse(encode(obs), encode(next_obs))
    inv_loss = F.cross_entropy(acts_pred, acts)
    obs_pred = forward(encode(obs), acts.view(-1, 1).float())
    fwrd_loss = F.mse_loss(obs_pred, encode(next_obs).detach())
    
    en_optim.zero_grad(); fwd_optim.zero_grad(); inv_optim.zero_grad()
    (inv_loss + fwrd_loss).backward()
    en_optim.step()
    fwd_optim.step()
    inv_optim.step()
    
    target_q = rewards + discount_fact * done * q_target(next_obs).max(dim=1)[0]
    target_q = target_q.view(-1, 1)
    q = q_net(obs).gather(1, acts.view(-1, 1).long())
    q_loss = F.smooth_l1_loss(q, target_q.detach())
    q_optim.zero_grad()
    q_loss.backward()
    q_optim.step()

if __name__ == '__main__':
    env = gym.make('Breakout-v0')
    q_net, q_target, encode, forward, inverse = QNet(), QNet(), Encoder(), \
                                         Forward_model(), Inverse_model()
    q_target.load_state_dict(q_net.state_dict())
    q_optim = optim.Adam(q_net.parameters(), lr=q_lr)
    en_optim = optim.Adam(encode.parameters(), lr=encoder_lr)
    fwd_optim = optim.Adam(forward.parameters(), lr=forward_lr)
    inv_optim = optim.Adam(inverse.parameters(), lr=inverse_lr)
                                              
    buffer = deque(maxlen=buffer_size)
    score, step = 0, 0
    epsilon, epsilon_decay = 0.6, 1-1e-5
    target_interval = 20
    
    for ep in range(EPISODES):
        obs = env.reset()
        obs = torch.tensor(obs).permute(2, 0, 1).float()
        done = False
        while not done:
            q_value = q_net(obs.unsqueeze(0))
            rand = random.random()
            if rand < epsilon:
                action = random.randint(0, 3)
            else:
                action = q_value.argmax().item()
            next_obs, _, done, info = env.step(action)
            next_obs = torch.tensor(next_obs).permute(2, 0, 1).float()
            
            obs_pred = forward(encode(obs.unsqueeze(0)), torch.tensor([[action]]).float())
            obs_ = encode(next_obs.unsqueeze(0))
            reward = reward_eta * F.mse_loss(obs_pred.squeeze(), obs_.squeeze()).item()
            buffer.append((obs.numpy(), action, reward, next_obs.numpy(), done))
            obs = next_obs
            step += 1
            score += reward
            epsilon *= epsilon_decay
            
        if len(buffer) > start_train:
            train((q_net, q_target, encode, forward, inverse), \
                  (q_optim, en_optim, fwd_optim, inv_optim), buffer)
        
        if ep%target_interval==0 and ep!=0:
            q_target.load_state_dict(q_net.state_dict())
            
        if ep%10==0 and ep!=0:
            print('episode:{}, step:{}, avg_score:{}, len_buffer:{}, epsilon:{}'.format(ep, step, \
                  score/10.0, len(buffer), epsilon))
            score = 0
    env.close()