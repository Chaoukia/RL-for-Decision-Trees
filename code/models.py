import torch.nn as nn
import torch.nn.functional as F

class DQNetwork(nn.Module):
    
    def __init__(self, input_size=25, out=8):
        """
        Description
        ---------------
        Constructor of DQNetwork class.
        
        Parameters
        ---------------
        input_size : Int, dimensionof the one-hot encoded representation of a state.
        out        : Int, output dimension, equal to the number of possible actions.
        fc_1       : nn.Linear, first fully connected layer.
        fc_2       : nn.Linear, second fully connected layer.
        output     : nn.Linear, output fully connected layer.
        """
        
        super(DQNetwork, self).__init__()
        
        self.fc_1 = nn.Linear(input_size, 32)
        self.fc_2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, out)
                
    def forward(self, x):
        """
        Description
        ---------------
        The forward pass.
        
        Parameters
        ---------------
        x : torch.tensor of dimension (batch_size, input_size)
        
        Returns
        ---------------
        torch.tensor of dimension (batch_size, out)
        """
        
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        return self.output(x)
    
    
class ActorCriticNetwork(nn.Module):
    
    def __init__(self, input_size=25, out=8):
        """
        Description
        ---------------
        Constructor of ActorCritic class.
        
        Parameters
        ---------------
        input_size    : Int, dimensionof the one-hot encoded representation of a state.
        out           : Int, Actor output dimension, equal to the number of possible actions.
        fc_1          : nn.Linear, first fully connected layer (common parameters between Actor and Critic).
        fc_2          : nn.Linear, second fully connected layer (common parameters between Actor and Critic).
        actor_output  : nn.Linear, actor output fully connected layer.
        critic_output : nn.Linear, critic output fully connected layer.
        """
        
        super(ActorCriticNetwork, self).__init__()
        
        self.fc_1 = nn.Linear(input_size, 32)
        self.fc_2 = nn.Linear(32, 32)
        
        self.actor_output = nn.Linear(32, out)
        self.critic_output = nn.Linear(32, 1)
        
        nn.init.constant_(self.actor_output.weight, 0)
        nn.init.constant_(self.actor_output.bias, 0)
                        
    def forward(self, x):
        """
        Description
        ---------------
        The forward pass.
        
        Parameters
        ---------------
        x : torch.tensor of dimension (batch_size, input_size)
        
        Returns
        ---------------
        value_output  : torch.tensor of dimension (batch_size, 1)
        policy_output : torch.tensor of dimension (batch_size, out)
        """
        
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        
        return self.critic_output(x), self.actor_output(x)
    
    
    