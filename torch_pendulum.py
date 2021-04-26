import torch
import numpy as np
import gym

class Pendulum:
    def __init__(self):
        
        self.d = 3
        self.ud = 1
        
        self.umax = 2
        
        self.env = gym.make('Pendulum-v0')
        
    def __del__(self):
        self.env.close()
            
    def get_data(self,Nsamples):
    
        states = []
        states_ = []
        actions = []
        for i in range(Nsamples):
            self.env.reset()
            for j in range(10):
                a = self.env.action_space.sample()
                s,r,_,_ = self.env.step(a) # take a random action
            states_.append(s)
            a = self.env.action_space.sample()
            s,r,_,_ = self.env.step(a) # take a random action
            actions.append(a)
            states.append(s)
        return states_,actions,states
        
    def dynamics(self,x,u):
        th = torch.atan2(x[:,1],x[:,0])
        if (th >= 0):
            th = th - 2*np.pi
        thdot = x[:,2]

        g = 10.
        m = 1.
        l = 1.
        dt = 0.05

        u = torch.clamp(u, -2, 2)[0]

        newthdot = thdot + (-3*g/(2*l) * torch.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = torch.clamp(newthdot, -8, 8)

        x = torch.cat([torch.cos(newth),torch.sin(newth),newthdot],dim=0).reshape(1,-1)
        return x
    
    def linear_dynamics(self,x,u):
        th = torch.atan2(x[:,1],x[:,0])
        if (th >= 0):
            th = th - 2*np.pi
        thdot = x[:,2]

        g = 10.
        m = 1.
        l = 1.
        dt = 0.05

        u = torch.clamp(u, -2, 2)[0]

        newthdot = thdot - (3*g/(2*l))*dt*th + np.pi + 3./(m*l**2)*u * dt
        newth = th + newthdot*dt
        newthdot = torch.clamp(newthdot, -8, 8)

        x = torch.cat([torch.cos(newth),torch.sin(newth),newthdot],dim=0).reshape(1,-1)
        return x
    
    def angle_normalize(self,x):
        return (((x+np.pi) % (2*np.pi)) - np.pi)

    def running_cost(self,x,u):
        th = torch.atan2(x[:,1],x[:,0])
        if (th >= 0):
            th = th - 2*np.pi
        return self.angle_normalize(th)**2 + .1*x[:,2]**2 + .001*(u[0]**2)
    
    def term_cost(self,x,u):
        return self.running_cost(x,u)
