import torch.nn as nn
import torch.nn.functional as F


class DQNetwork(nn.Module):
    
    def __init__(self, input_size=25, out=8):
        """
        Description
        ---------------
        Constructor of Deep Q-network class.
        
        Parameters
        ---------------
        input_size : Int, the one-hot encoding representation dimension.
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


















