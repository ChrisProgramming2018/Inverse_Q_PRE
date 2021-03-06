import gym
import random
import torch
import json
import argparse
import numpy as np
from dqn_agent import DQNAgent
from agent_iql import Agent
from helper import FrameStack
import time 




def main(args):
    with open (args.param, "r") as f:
        config = json.load(f)
    
    env = gym.make('MontezumaRevenge-v0')
    env.seed(args.seed)
    env  = FrameStack(env, args)

    print('State shape: ', env.observation_space.shape)
    print("use agent ", args.agent)
    agent = DQNAgent(state_size=200, action_size=7, config=config)
    agent_r = Agent(state_size=200, action_size=7,  config=config)
    #agent_r.load("models-28_11_2020_13:28:57/51800-")
    #agent_r.load("models-29_11_2020_08:35:01/80500-")
    agent_r.load("models-29_11_2020_20:45:22/7000-")
    env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True,force=True)
    #agent.qnetwork_local.load_state_dict(torch.load('checkpoint-score80.47156817885116_epi_125.pth'))
    agent.qnetwork_local.load_state_dict(torch.load('models/checkpoint_q-{}.pth'.format(args.agent)))
    agent.encoder.load_state_dict(torch.load('models/checkpoint_e-{}.pth'.format(args.agent)))
    n_episodes = 10
    max_t = 500
    eps = 0
    action_size = 7
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        env.seed(random.randint(0,30))
        score = 0
        for t in range(max_t):

            # action = agent.act(state, eps)
            action = agent_r.act(state)
            
            print(action)
            #action = random.choice(np.arange(action_size))
            next_state, reward, done, _ = env.step(action)
            score += reward
            time.sleep(0.01)
            state = next_state
            env.render()
            if done:
                break
        print("Episode {}  Reward {} Steps {}".format(i_episode, score, t))
        env.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', default="param.json", type=str)
    parser.add_argument('--size', default=84, type=int)
    parser.add_argument('--history_length', default=3, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--agent', default=20, type=int)
    parser.add_argument('--seed', default=20, type=int)
    arg = parser.parse_args()
    main(arg)
