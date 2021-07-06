import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque

#Hyperparameters
iteration = 1000
update_epi = 20
discount_factor = 0.98
train_interval = 10
buffer_size, start_train = 10000, 16
batch_size = 16
target_update = 0.995
noise_size = 12
goal_tolerance = 0.5
gen_num, sample_num = 1, 2
smple_start = 2
max_step = 100
r_min, r_max = 0.05, 0.95
q_lr = 0.0005
mu_lr = 0.0001
gen_lr = 0.0005
disc_lr = 0.0005
state_space, action_space = 24, 4 #action_space is continuous


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(noise_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.goal = nn.Linear(256, state_space)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.goal(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(state_space, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 1)
    
    def forward(self, goal):
        x = F.relu(self.fc1(goal))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #least-square loss
        return self.output(x)
    
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.obs = nn.Linear(state_space, 128)
        self.goal = nn.Linear(state_space, 128)
        self.act = nn.Linear(action_space, 128)
        self.fc1 = nn.Linear(384, 720)
        self.fc2 = nn.Linear(720, 720)
        self.q = nn.Linear(720, 1)
    
    def forward(self, x, goal, a):
        x = F.relu(self.obs(x))
        z = F.relu(self.goal(goal))
        a = F.relu(self.act(a))
        x = torch.cat([x, z, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)

class ActionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.obs = nn.Linear(state_space, 256)
        self.goal = nn.Linear(state_space, 256)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.action = nn.Linear(512, action_space)
    
    def forward(self, x, goal):
        x = F.relu(self.obs(x))
        z = F.relu(self.goal(goal))
        x = torch.cat([x, z], dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        a = torch.tanh(self.action(x))
        return a

def train(q, q_target, mu, mu_target, buffer, q_optimizer, mu_optimizer):
    mini_batch = random.sample(buffer, batch_size)
    obs, goals, acts, rewards, next_obs, done = [], [], [], [], [], []
    for samples in mini_batch:
        s, g, a, r, s_, d = samples
        d = 0.0 if d else 1.0
        obs.append(s); goals.append(g); acts.append(a); rewards.append(r); 
        next_obs.append(s_); done.append(d)
    
    obs, goals, acts, rewards, next_obs, done = torch.tensor(obs).float(), \
    torch.tensor(goals).float(), torch.tensor(acts).float(), \
    torch.tensor(rewards).float(), torch.tensor(next_obs).float(), \
    torch.tensor(done)
    
    target_a = mu_target(next_obs, goals)
    target_q = rewards.view(-1, 1) + discount_factor * done.view(-1, 1) \
                                     * q_target(next_obs, goals, target_a)
    
    q_loss = F.smooth_l1_loss(q(obs, goals, acts.view(-1, action_space)), target_q.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()
    mu_obj = -q(obs, goals, mu(obs, goals)).mean()
    mu_optimizer.zero_grad()
    mu_obj.backward()
    mu_optimizer.step()

#Implementation of soft-update
def soft_update(t_net, net, target_ratio):
    for t_param, param in zip(t_net.parameters(), net.parameters()):
        t_param.data.copy_(t_param.data*target_ratio + param.data*(1-target_ratio))
    
def update_policy(env, episode, goals, buffers):
    global qnet, q_target, munet, mu_target
    global q_optimizer, mu_optimer
    goal_label = []
    for idx, goal in enumerate(goals):
        score = 0.0
        for ep in range(episode):
            done = False
            obs = env.reset()
            step = 0
            while not done:
                a = munet(torch.tensor(obs).unsqueeze(0).float(), goal.unsqueeze(0))
                noise = torch.randn(4) * 0.3
                action = torch.clamp(a+noise, -1, 1)[0].detach().numpy()
                next_obs, _, _, info = env.step(action)
                dist = F.pairwise_distance(torch.tensor(next_obs).unsqueeze(0).float(), \
                                           goal.unsqueeze(0))
                reward, done = 0.0, False
                if dist.item() < goal_tolerance:
                    reward, done = 1.0, True
                if step > max_step: done = True 

                buffers[idx].append((obs, goal.detach().numpy(), action, \
                                     reward/10.0, next_obs, done))
                score += reward
                step += 1
                obs = next_obs
                if step%train_interval==0 and len(buffers[idx]) > start_train:
                    train(qnet, q_target, munet, mu_target,\
                          buffers[idx], q_optimizer, mu_optimizer)
                    soft_update(q_target, qnet, target_update)
                    soft_update(mu_target, munet, target_update)
        
        print('epsidoes:{}, goal_success:{}'.format(ep, score/episode))
        prob = score/episode
        label =  1 if prob >= r_min and prob <= r_max else 0
        goal_label.append(label)
    return buffers, goal_label

def train_gan(goals, label):
    global generator, discrim, gen_optimizer, disc_optimizer
    logit = discrim(torch.stack(goals, dim=0))
    label = torch.tensor(label).unsqueeze(1)
    
    data_loss = label*(logit - 1.0).pow(2) + (1-label)*(logit + 1.0).pow(2)
    z = torch.randn(label.size(0), noise_size)
    gen_loss = (discrim(generator(z).detach()) + 1).pow(2)
    discrim_loss = (data_loss + gen_loss).mean()
    disc_optimizer.zero_grad()
    discrim_loss.backward()
    disc_optimizer.step()
    
    z = torch.randn(8, noise_size)
    gen_loss = discrim(generator(z)).pow(2).mean()
    gen_optimizer.zero_grad()
    gen_loss.backward()
    gen_optimizer.step()
    
if __name__ == '__main__':
    env = gym.make('BipedalWalkerHardcore-v3')
    qnet, q_target, munet, mu_target, generator, discrim = QNet(), QNet(), \
    ActionNet(), ActionNet(), Generator(), Discriminator()
    q_target.load_state_dict(qnet.state_dict())
    mu_target.load_state_dict(munet.state_dict())
    q_optimizer, mu_optimizer = optim.Adam(qnet.parameters(), lr=q_lr), \
                                optim.Adam(munet.parameters(), lr=mu_lr)
    gen_optimizer = optim.Adam(generator.parameters(), lr=gen_lr)
    disc_optimizer = optim.Adam(discrim.parameters(), lr=disc_lr)
    buffers, goals_old = {}, []
    
    for i in range(iteration):
        gen_goal = list(generator(torch.randn(gen_num, noise_size)))
        buffers.update({g:deque(maxlen=buffer_size) for g in gen_goal})
        #generate goals-list with new goal and sampled goal
        goals = gen_goal
        if len(goals_old) > smple_start:
            smple_goal = random.sample(goals_old, sample_num)
            goals += smple_goal
        goal_buffers = [buffers[g] for g in goals]
        #train policy with goals-list
        goal_buffers, goal_label = update_policy(env, update_epi, goals, goal_buffers)
        buffers.update({g:goal_buffers[idx] for idx, g in enumerate(goals)})
        #goal evaluation
        for idx, g in enumerate(goals):
            if goal_label[idx]:
                goals_old.append(g)
        train_gan(goals, goal_label)
    env.close()