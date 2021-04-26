import torch
import torch.nn as nn
from torch.distributions import Normal, OneHotCategorical, MultivariateNormal
from torch.autograd import Variable # storing data while learning

class Decoder(nn.Module):
    
    def __init__(self,hidden_dim=3,latent_dim=3,d=3):
        super().__init__()
        
        self.decoder = nn.Sequential(
                        nn.Linear(hidden_dim,latent_dim),
                        nn.ReLU(),
                        nn.Linear(latent_dim,latent_dim),
                        nn.ReLU(),
                        nn.Linear(latent_dim,d))

    def forward(self, x):
        params = self.decoder(x)
        return params
    
class Encoder(nn.Module):

    def __init__(self,input_dim=3,latent_dim=3,hidden_dim=3):
        super().__init__()
        
        self.encoder = nn.Sequential(
                        nn.Linear(input_dim,latent_dim),
                        nn.ReLU(),
                        nn.Linear(latent_dim,latent_dim),
                        nn.ReLU(),
                        nn.Linear(latent_dim,hidden_dim))

    def forward(self, x):
        params = self.encoder(x)
        return params
   

class FCN(nn.Module):
    
    def __init__(self,latent_dim=16,d=2,ud=1):
        super(FCN,self).__init__()

        self.model = Encoder(d+ud,latent_dim,d)
        
    def forward(self,x_,u):
        return self.model(torch.cat([x_,u],dim=1))
    
    def dynamics(self,x_,u):
        return self.forward(x_,u)
        
    
    def loss_fn(self,recon_x,x):
        return torch.sum((recon_x - x)*(recon_x - x))
    

# MVNormalNetwork
class MVNormalNetwork(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()
        
        self.mean = nn.Linear(latent_dim,latent_dim)
       
        self.sc = nn.Linear(latent_dim,latent_dim)
        
    def forward(self, x):
        
        mean = self.mean(x)
        sc = self.sc(x)
        
        return mean, torch.diag_embed(torch.exp(sc))

    
class LTAutoEncoder(nn.Module):
    
    def __init__(self,d=3,latent_dim=3,hidden_dim=2,ud=1):
        super().__init__()
        
        self.decoder = Decoder(hidden_dim,latent_dim,d)
        self.encoder = Encoder(d,latent_dim,hidden_dim)
        
        self.mv = MVNormalNetwork(hidden_dim)
        
        self.A = torch.nn.Parameter(torch.randn(hidden_dim,hidden_dim))
        self.B = torch.nn.Parameter(torch.randn(ud,hidden_dim))
        
        self.Qdiag = torch.nn.Parameter(torch.randn(hidden_dim))
        
        self.reconloss = nn.MSELoss(reduction='mean')
        
    def forward(self, x):
        
        latent = self.encoder(x)
        mu, sc = self.mv(latent)
        
        z = self.reparameterize(mu, sc)
        
        return self.decoder(z), mu, sc
        
    def latent_dynamics(self,mu_0,sc_0,action):
        
        mu_1 = torch.matmul(mu_0,self.A) + torch.matmul(action,self.B)
        
        cov_0 = torch.matmul(sc_0,sc_0.transpose(1,2))
        
        cov_1 = torch.matmul(torch.matmul(self.A,cov_0),self.A.t()) + torch.diag(torch.exp(self.Qdiag))
        
        sc_1 = torch.cholesky(cov_1)
        
        return mu_1, sc_1
    
    def dynamics(self,x,u):
        
        latent = self.encoder(x)
        mu_0, sc = self.mv(latent)
        
        mu_1 = torch.matmul(mu_0,self.A) + torch.matmul(u,self.B)
        
        return self.decoder(mu_1)
    
    def loss(self,x_0,x_1,action):
        
        mu_0, sc_0 = self.mv(self.encoder(x_0))
        mu_1, sc_1 = self.mv(self.encoder(x_1))
        y = self.decoder(self.reparameterize(mu_1,sc_1))
        
        muhat_1, schat_1 = self.latent_dynamics(mu_0,sc_0,action)
        
        q = MultivariateNormal(mu_1, scale_tril=sc_1)
                
        p = MultivariateNormal(muhat_1,scale_tril=schat_1)
        
        kle = torch.distributions.kl_divergence(q, p).mean()
        
        recon = self.reconloss(y,x_1)
        
        return recon + kle
    
    def reparameterize(self, mu, L):
        
        d = mu.size()
        
        eps = Variable(torch.FloatTensor(d).normal_())

        return torch.squeeze(torch.matmul(L,torch.unsqueeze(eps,dim=2))) + mu
    
