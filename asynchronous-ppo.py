import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import threading as T

#Hyperparameters
EPISODES = 10000
learning_rate = 0.0002
discount_factor = 0.98
train_interval = 20
num_agents = 3
lmbda = 0.5
epsilon = 0.2
#Note:Unlike single-agent PPO, train-iteration is fixed at 1

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.p = nn.Linear(128, 2)
        self.value = nn.Linear(128, 1)
    
    def pi(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        prob = F.softmax(self.p(x), dim=1)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.value(x)

def mini_batch(samples):
    obs, acts, probs, rewards, next_obs, done = [], [], [], [], [], []
    for transition in samples:
        s, a, p, r, s_, d = transition
        d = 0.0 if d else 1.0
        obs.append(s); acts.append(a); probs.append(p), rewards.append(r)
        next_obs.append(s_), done.append(d)
    
    return torch.tensor(obs).float(), torch.tensor(acts), torch.tensor(probs).float(), \
    torch.tensor(rewards).float(), torch.tensor(next_obs).float(), torch.tensor(done)
    
def train(net, samples, global_optimizer):
    obs, acts, probs, rewards, next_obs, done = mini_batch(samples)
    target = rewards.view(-1, 1) + discount_factor * net.v(next_obs) * done.view(-1, 1)
    td = target - net.v(obs)
    #Implementation of GAE(Generalized Advantage Estimation)
    advantage, R = [], 0.0
    for delta in torch.flip(td, dims=[0, 1]):
        R = delta + discount_factor * lmbda * R
        advantage.append(R)
    advantage.reverse()
    advantage = torch.tensor(advantage).float().unsqueeze(1)
    
    pi_a = net.pi(obs).gather(1, acts.view(-1, 1))
    ratio = torch.exp(torch.log(pi_a) - torch.log(probs.view(-1, 1)).detach()) 
    clipped = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    
    p_loss = -torch.min(ratio*advantage, clipped*advantage)
    v_loss = F.smooth_l1_loss(net.v(obs), target.detach())
    loss = (p_loss + v_loss).mean()
    
    global_optimizer.zero_grad()
    loss.backward()
    for global_param, local_param in zip(global_net.parameters(), net.parameters()):
        global_param._grad = local_param.grad
    global_optimizer.step()

def agent(rank):
    env = gym.make('CartPole-v1')
    net = Network() #define local network
    net.load_state_dict(global_net.state_dict())
    global_optimizer = optim.Adam(global_net.parameters(), lr=learning_rate)
    samples, score, step = [], 0.0, 0
    
    for ep in range(EPISODES):
        obs = env.reset()
        done = False
        while not done:
            prob = net.pi(torch.tensor(obs).unsqueeze(0).float())
            prob_ = Categorical(prob)
            action = prob_.sample().item()
            next_obs, reward, done, info = env.step(action)
            samples.append((obs, action, prob[0][action], reward/100.0, next_obs, done)) 
            score += reward
            step += 1
            obs = next_obs
            
            if step%train_interval==0:
                train(net, samples, global_optimizer)
                net.load_state_dict(global_net.state_dict())
                samples = []
        
        if ep%10==0 and ep!=0:
            print('agent_num:{}, episode:{}, avg_score:{}'.format(rank,ep,score/10.0))
            score = 0.0
    env.close()

if __name__ == '__main__':
    global_net = Network()
    agents = []
    for rank in range(num_agents):
        actor = T.Thread(target=agent, args=(rank,))
        actor.start()
        agents.append(actor)
    for t in agents:
        t.join()