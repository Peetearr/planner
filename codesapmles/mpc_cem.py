import os
import numpy as np
import gym
from hand_imitation.env.environments.shapenet_relocate_env import SHAPENETRelocate
import numpy as np
import pickle
import copy
import argparse

class MPC:
    def __init__(self, env, plan_horizon = 8, popsize = 200, num_elites = 10, max_iters = 2, use_mpc = True):
        """
        :param env:
        :param plan_horizon: 
        :param popsize: Population size
        :param num_elites: CEM parameter
        :param max_iters: CEM parameter
        :param use_mpc: Whether to use only the first action of a planned trajectory
        """
        self.env = env
        self.use_mpc = use_mpc
        self.plan_horizon = plan_horizon
        self.max_iters = max_iters
        self.popsize = popsize
        self.num_elites = num_elites
        self.action_dim = 30
        self.ac_ub, self.ac_lb = 1, -1 
        self.reset()

    def reset(self):
        self.mean = np.zeros((self.plan_horizon * self.action_dim))
        self.std = 0.5 * np.ones((self.plan_horizon * self.action_dim))
        
    def predict_next_state_gt(self, states, actions):
        """ Given a list of state action pairs, use the ground truth dynamics to predict the next state"""
        next_states = []
        rewards = []
        for i in range(len(states)):
            next_state, reward, info = self.env.mpc_step(states[i], actions[i], t)
            next_states.append(next_state)
            rewards.append(reward)
        return next_states, rewards

    def cem_optimize(self, state):
        mean = self.mean.copy()
        std = self.std.copy()
        #next_state = state.copy()
        for i in range(self.max_iters):
            return_list=[]
            action_list=[]


            sample_mean=np.expand_dims(mean,0).repeat(self.popsize,axis=0)
            sample_std=np.expand_dims(std,0).repeat(self.popsize,axis=0)

            actions=np.random.normal(sample_mean,sample_std)

            next_states=[]#np.expand_dims(state,0).repeat(self.popsize,axis=0)
            for ii in range(self.popsize):
                next_states.append(state)
            rewards=0
            for T in range(self.plan_horizon):
                next_states,reward=self.predict_next_state_gt( next_states, actions[:,T*30:(T+1)*30])
                rewards=rewards+np.array(reward)
                
                
            index=np.argsort(np.array(rewards))[-self.num_elites:]
            select_actions=actions[index]
            mean=np.mean(select_actions,axis=0)
            std=np.std(select_actions,axis=0)

                
        self.mean=mean
        self.std=std
        return mean, std

    def act(self, state, t):
        """
        Use model predictive control to find the action give current state.

        Arguments:
          state: current state
          t: current timestep
        """
        if self.use_mpc == False:
            if t % self.plan_horizon == 0:
                self.reset()
                mean,std=self.cem_optimize(state)
                action=mean[:30]
                self.temp_actions=mean
                return action
            else:
                ii= (t % self.plan_horizon)
                action= self.temp_actions[ii*30:(ii+1)*30]
                pass
            return action 
        else:
            
            mean,std=self.cem_optimize(state)
            action=mean[:30]
            
            self.mean=np.concatenate([mean[30:],np.zeros(30)])
            self.std=np.concatenate([std[30:],np.ones(30)*0.5])
            return action

if __name__ == "__main__":
    import time
    #generate data
    category='bottle'
    
    obj_poses_dict=np.load(f'./data/new_shapenet_{category}_poses_scale_1.npy',allow_pickle=True).item()
    joints_dict=np.load(f'./data/new_joints_shapenet_{category}_scale_1.npy',allow_pickle=True).item()

    
    object_names = list(obj_poses_dict.keys())
    joint2world=np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
    


    
    saved_actions_dict={}
    need=[2,3,7,11,15,20,25]
    count=0
    for object_name in object_names:
        #if count>0:
        #    break

        obj_poses=obj_poses_dict[object_name]
        saved_actions_dict[object_name]=[]

        for ii in range(len(obj_poses)):
            temp=[]
            obj_pose=obj_poses[ii]
            target_hand_joints_list=joints_dict[object_name][ii]

            for iii in range(len(target_hand_joints_list)):
                print(object_name,ii,iii)
                target_hand_joints=target_hand_joints_list[iii]
                
                #print(target_hand_joints)
                #target_robot_pos
                
                env = SHAPENETRelocate(has_renderer=True, object_name=object_name, category=category, friction=(1, 0.5, 0.01), object_scale=1)
                env.seed(0)
                np.random.seed(0)
                obs = env.reset()
                cem_mpc = MPC(env,use_mpc=True)
                states = []
                actions = []
                next_states = []
                
                state = env.sim.get_state()
                
                state[1][-7:]=obj_pose
                env.sim.set_state(state)

                
                env.set_target_hand_joints(target_hand_joints+[0,0,0.2],obj_pose)
                distance_list=[]
                for t in range(60):
                    #print(iii,t)
                    if t==15:
                        env.set_target_hand_joints(target_hand_joints,obj_pose)                        
                        

                    state = env.sim.get_state()
                    action = cem_mpc.act(state, t)
                    #action=saved_actions[t]
                    states.append(state)
                    actions.append(action)
                    obs, reward, done, info = env.step(action,t)
                    #print(reward)
                    #print(env.sim.get_state()[1][30:33])
                    #print(env.data.body_xpos[[2]])                
                    #time.sleep(0.1)
                    env.render()
                    
                    sign=np.linalg.norm(env.data.site_xpos[[8,12,16,21,25]]-target_hand_joints[[17,18,19,20,16]])
                    

                    distance_list.append(sign)
                    std=np.std(distance_list[-5:])

                    if (sign<0.1 and std<0.005 and reward>-0.8) or (sign<0.06 and reward>-0.8) :
                       print('save')
                       break
                    

                env.close()

                if t==59:
                    continue
                else:
                    temp.append(np.array(actions))
            saved_actions_dict[object_name].append(temp)
        
            