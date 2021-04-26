import torch

class OptControl:
    
    def __init__(self,dynamics,running_cost,term_cost,u_dim=1,umax=2,horizon=10,lr=1e-2):
        
        self.dynamics = dynamics
        self.term_cost = term_cost
        self.running_cost = running_cost
        self.horizon = horizon
        self.u = torch.zeros(1,self.horizon)
        self.umax = 2
        self.lr = lr

    def cost(self,x,u):
        cost = []
        states = [x]
        for j in range(self.horizon-1):
            states.append(self.dynamics(states[-1].reshape(1,-1),self.umax*torch.tanh(u[:,j]).reshape(1,-1)))
            cost.append(self.running_cost(states[-1].reshape(1,-1),self.umax*torch.tanh(u[:,j]).reshape(1,-1)))
            
        return torch.sum(torch.stack(cost))+self.term_cost(states[-1],u[:,-1]), states, cost
    
    def minimize(self,xin,Nsteps=10):
        u = torch.nn.Parameter(torch.roll(self.u,shifts=-1,dims=1))
        optimizer = torch.optim.Adam([u], lr=self.lr)
        for i in range(Nsteps):
            loss,states,cost = self.cost(xin,u)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.u = u
        return self.umax*torch.tanh(u),states,loss.item(),cost
