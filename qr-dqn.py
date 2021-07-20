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
num_support = 20
k = 1.0 #for huber loss
state_space, action_space = 8, 4

#============================ base formula ============================
#make culminative distribution(tau-1 .... tau-n) from uniform probablity
tau_prob = [n/num_support for n in range(num_support+1)] 
#get middle of two-taus(which is *unique minimizer* of wasserstein distance)
mid_prob = [(tau_prob[i] + tau_prob[i+1])/2 for i in range(num_support)]

class Quantile(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(state_space, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.acts = [nn.Linear(256, num_support) for _ in range(action_space)]
    
    def forward(self, x):
        '''
        network-input: state
        output:        q-distribution for each action -> Z(s, .)
        output-shape:  (actions, *batch_size*, supports)
        '''
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = [self.acts[i](x) for i in range(action_space)]
        return value

def make_batch(buffer):
    '''
    Make batch of train-samples by sampling from the buffer
    '''
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
    '''
    Train network by samples from buffer
    
    In this function, 
    next_supports means *q-distribution* over next-states
    supports means *q-distribution* over states
    '''
    obs, acts, rewards, next_obs, done = make_batch(buffer)
    next_supports = target_net(next_obs)
    #next_supports(=list)'s shape is (*act_space*, batch_size, num_support)
    next_supports = torch.stack(next_supports, dim=1) #convert to tensor
    #now, shape is (*batch_size*, act_space, num_support)
    
    next_q = (1/num_support) * torch.sum(next_supports, dim=2) #get Q-value from dist.
    #next_q(expectation over support)'s shape is (batch_size, act_space)
    max_acts = next_q.argmax(dim=1) 
    #max_acts'shape is just (batch_size,)
    
    #============= get next-supports of optimal actions => Z(s', a*) ==============
    max_quantile = [next_supports[idx][max_a] for idx, max_a in enumerate(max_acts)]
    max_quantile = torch.stack(max_quantile, dim=0) #just convert to tensor
    #max_quantile's shape is (batch_size, *num_support*) (actions were reduced)
    
    target_supports = rewards.view(-1, 1) + discount_factor * max_quantile
    
    supports = torch.stack(net(obs), dim=1) #supports over states
    
    #============= get supports of action => Z(s, a) ==============
    supports_a = [supports[idx][a] for idx, a in enumerate(acts)]
    supports_a = torch.stack(supports_a, dim=0) #just convert to tensor
    
    #============== Quantile Regression Loss Calculation =================
    target_supports = target_supports.view(batch_size, -1, 1).expand(-1, num_support, num_support).detach()
    supports_a = supports_a.view(batch_size, 1, -1).expand(-1, num_support, num_support)
    diff = target_supports - supports_a
    
    #Huber loss calculation
    soft_diff = torch.where(torch.abs(diff)<=k, 0.5*torch.pow(diff, 2), \
                            k*(torch.abs(diff) - 0.5*k))
    s_diff1 = torch.tensor(mid_prob) * soft_diff
    s_diff2 = (1 - torch.tensor(mid_prob)) * soft_diff
    error = torch.where(diff>=0, s_diff1, s_diff2)
    loss = torch.sum(error) / (batch_size * num_support)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    net, target_net = Quantile(), Quantile()
    target_net.load_state_dict(net.state_dict())
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, eps=0.01/32)
    
    buffer = deque(maxlen=buffer_size)
    score, step = 0, 0
    epsilon, epsilon_decay = 0.2, 1-5e-6
    target_interval = 20
    
    for ep in range(EPISODES):
        obs = env.reset()
        done = False
        while not done:
            quantiles = net(torch.tensor(obs).unsqueeze(0).float()) 
            #shape: (action_space, 1, num_supports)
            quantiles = torch.stack(quantiles, dim=1) #shape: (1, action_space, num_supports)
            qvals = (1/num_support) * torch.sum(quantiles, dim=2) #mean over supports
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
            print('episode:{}, step:{}, avg_score:{}, len_buffer:{}, epsilon:{}'.format(ep, step, \
                  score/10.0, len(buffer), epsilon))
            score = 0
    env.close()