import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import random
import itertools
from collections import deque
from torch.distributions import Categorical

#Note: single-thread version, support trpo-update
#Hyperparameters
EPISODES = 10000
learning_rate = 0.0002
discount_factor = 0.98
train_interval = 10
replay_iter = 8
buffer_len, start_train = 20000, 500
is_clipping = 1.2
trpo_delta = 1.0
avgnet_ratio = 0.995

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.policy = nn.Linear(128, 2)
        self.qval = nn.Linear(128, 2)
    
    def p(self, x):
        x = F.relu(self.fc1(x))
        self.pi = F.relu(self.fc2(x))
        self.pi.retain_grad()
        prob = F.softmax(self.policy(self.pi), dim=1)
        return prob
    
    def q(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.qval(x)

def mini_batch(data):
    obs, acts, probs, rewards, next_obs, done = [], [], [], [], [], []
    for transition in data:
        s, a, p, r, s_, d = transition
        obs.append(s); acts.append(a); probs.append(p); rewards.append(r)
        next_obs.append(s_), done.append(d)
    
    return torch.tensor(obs).float(), torch.tensor(acts), \
           torch.stack(probs, dim=0).float(), torch.tensor(rewards).float(),\
           torch.tensor(next_obs).float(), torch.tensor(done)
    
def train_process(net, avg_net, samples, optimizer):
    obs, acts, old_probs, rewards, next_obs, done = samples
    acts, rewards = acts.view(-1, 1), rewards.view(-1, 1)
    final_q, final_p = net.q(next_obs[-1].unsqueeze(0)), net.p(next_obs[-1].unsqueeze(0))
    final_v = torch.sum(final_q * final_p, dim=1)
    qval = net.q(obs)
    current_p = net.p(obs)
    avg_p = avg_net.p(obs)
    value = torch.sum(qval*current_p, dim=1, keepdim=True)
    
    act_q = qval.gather(1, acts)
    ratio = torch.exp(torch.log(current_p) - torch.log(old_probs))
    ret_ratio = torch.min(torch.tensor(1.0), ratio.gather(1, acts))
    policy_ratio = torch.min(torch.tensor(is_clipping), ratio.gather(1, acts))
    
    ret_q = []
    R = final_v if not done[-1] else torch.tensor([0.0])
    for idx, r in enumerate(torch.flip(rewards, [0, 1])):
        R = r + discount_factor * R
        ret_q.append(R)
        R = ret_ratio[-1-idx]*(R - act_q[-1-idx]) + value[-1-idx]
    ret_q.reverse()
    ret_q = torch.stack(ret_q, dim=0)
    
    p_obj1 = policy_ratio.detach() * torch.log(current_p.gather(1, acts)) * \
             (ret_q - value).detach()
    p_obj2 = 0
    for a in range(2):
        coeff = torch.max(torch.tensor(0), 1-is_clipping/ratio[:, a]).view(-1, 1)
        a_prob, a_qval = current_p[:, a].view(-1, 1), qval[:, a].view(-1, 1)
        p_obj2 += (coeff*a_prob).detach() * torch.log(a_prob) * (a_qval - value).detach()
    
    policy_obj = (p_obj1 + p_obj2).mean()
    
    g = autograd.grad(policy_obj, net.pi, retain_graph=True)[0]
    kld = F.kl_div(avg_p.detach(), current_p)
    k = autograd.grad(kld, net.pi, retain_graph=True)[0]
    #trust-region update
    k_norm = torch.linalg.norm(k, dim=1).view(-1, 1, 1)**2
    g_, k_ = g.unsqueeze(2), k.unsqueeze(1)
    solve = (torch.bmm(k_, g_) - trpo_delta) / k_norm
    new_g = g - torch.max(torch.tensor(0), solve.view(-1, 1))*k

    q_loss = F.smooth_l1_loss(act_q, ret_q.detach())
    optimizer.zero_grad()
    net.policy.weight._grad = autograd.grad(-policy_obj, net.policy.weight, retain_graph=True)[0]
    net.pi.backward(-new_g)
    q_loss.backward()
    optimizer.step()
    
def train(net, avg_net, online_sample, buffer, optimizer):
    train_process(net, avg_net, mini_batch(online_sample), optimizer)
    
    if len(buffer) > start_train:
        for _ in range(replay_iter):
            key = random.randint(0, len(buffer)-train_interval)
            replay_sample = itertools.islice(buffer, key, key+train_interval)
            train_process(net, avg_net, mini_batch(replay_sample), optimizer)

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    net, avg_net = Network(), Network()
    avg_net.load_state_dict(net.state_dict())
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    buffer = deque(maxlen=buffer_len)
    samples, score, step = [], 0.0, 0
    
    for ep in range(EPISODES):
        obs = env.reset()
        done = False
        while not done:
            prob = net.p(torch.tensor(obs).unsqueeze(0).float())
            prob_ = Categorical(prob)
            action = prob_.sample().item()
            next_obs, reward, done, info = env.step(action)
            data = (obs, action, prob[0], reward/100.0, next_obs, done)
            samples.append(data)
            buffer.append(data)
            score += reward
            step += 1
            obs = next_obs
            
            if step%train_interval==0:
                train(net, avg_net, samples, buffer, optimizer)
                for a_param, param in zip(avg_net.parameters(), net.parameters()):
                    a_param.data.copy_(a_param.data*avgnet_ratio + param.data*(1-avgnet_ratio))
                samples = []
        
        if ep%10==0 and ep!=0:
            print('episode:{}, num_train:{}, avg_score:{}'.format(ep, \
                   step//train_interval, score/10.0))
            score = 0.0
    env.close()