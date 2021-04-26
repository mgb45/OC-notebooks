import torch

class H5Dataset(torch.utils.data.Dataset):

    def __init__(self, states_,actions,states,mu=0,sd=1):
        super(H5Dataset, self).__init__()
        
        self.states = torch.from_numpy((states-mu)/sd).float()
        self.states_ = torch.from_numpy((states_-mu)/sd).float()
        self.actions = torch.from_numpy(actions).float()
        
    def __getitem__(self, index): 
            
        return self.states_[index], self.actions[index], self.states[index]

    def __len__(self):
        return self.states.shape[0]