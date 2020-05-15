from utils import *


class Environment:
    """
    Description
    --------------
    Class representing the environment, it can generate data points that start each episode,
    keep track of the current state, return the reward of an action taken at the current state,
    and transition to the next corresponding state.
    """
    
    def __init__(self, generator, rewards_queries, r_plus=5, r_minus=-5, split=3):
        """
        Description
        --------------
        Constructor of class Environment.
        
        Parameters & Attributes
        --------------
        generator       : Dict, - keys   : Feature variables.
                                - values : List of probability masses of each category of the corresponding feature.
        rewards_queries : Dict, - keys   : Feature variables.
                                - values : Reward of querying the value of the corresponding feature.
        r_plus          : Int, reward of a correct report (default=5).
        r_minus         : Int, reward of an incorrect report (default=-5).
        split           : Int, the split point we use to define our concept.
        d               : Int, the number of feature variables.
        data_point      : List of length d, the data point starting the episode.
        label           : Boolean, the true label of data_point.
        state           : Object of class State, the current state.
        done            : Boolean, whether the episode is finished or not.
        """
        
        self.generator = generator
        self.categories = [len(v) for v in self.generator.values()]
        self.d = len(self.categories)
        self.rewards_queries = rewards_queries
        self.r_plus = r_plus
        self.r_minus = r_minus
        self.split = split
        
    def generate(self):
        """
        Description
        --------------
        Generate a data point.
        
        Parameters
        --------------
        
        Returns
        --------------
        List with the values of each feature, it represents the data point.
        """
            
        return [np.random.choice(self.categories[i], p=self.generator[i]) for i in range(self.d)]
    
    def concept(self, data_point):
        """
        Description
        --------------
        Define the concept labeling the data points. we can define it as a decision tree for example.
        
        Parameters
        --------------
        data_point : List of length d, the data point to label.
        
        Returns
        --------------
        Int in {0, 1}, the label of the data point.
        """
        
        label = True
        i = 0
        while label and i <= self.d-1:
            if data_point[i] >= self.split:
                label = False
                
            i += 1
            
        return label
    
    def reset(self, data_point=None):
        """
        Description
        --------------
        Reset the environment to start a new episode. If data_point is specified, start the episode from it,
        otherwise generate it.
        
        Parameters
        --------------
        data_point : List of length d, the data point to label (default=None).
        
        Returns
        --------------
        """
        
        self.data_point = self.generate() if data_point is None else data_point
        self.label = self.concept(self.data_point)
        self.state = State([np.NaN for i in range(self.d)], categories=self.categories)
        self.done = False
    
    def step(self, action):
        """
        Description
        --------------
        Interract with the environment through an action taken at the current state.
        
        Parameters
        --------------
        action : Int in {0, ..., d-1, d, d+1}, 
                 - 0, ..., d-1 represent query actions.
                 - d, d+1 represent report actions.
        
        Returns
        --------------
        reward     : Int, the reward of taking this action at the current state.
        next_state : Object of class State, the next state.
        done       : Boolean, whether the episode is finished or not.
        """
        
        # Treating query actions.
        if action <= self.d-1:
            reward = self.rewards_queries[action]
            values = self.state.values
            values[action] = self.data_point[action] # Reveal the value of the queried feature in the data point.
            self.state = State(values)
            
        # Treating report actions.
        else:
            reward = self.r_plus if (action%self.d) == self.label else self.r_minus
            self.done = True
            
        return reward, self.state, self.done

    

    
class EnvironmentDQN:
    """
    Description
    --------------
    Class representing the environment, it can generate data points that start each episode,
    keep track of the current state, return the reward of an action taken at the current state,
    and transition to the next corresponding state.
    """
    
    def __init__(self, generator, rewards_queries, encoder, r_plus=5, r_minus=-5, split=3):
        """
        Description
        --------------
        Constructor of class Environment.
        
        Parameters & Attributes
        --------------
        generator       : Dict, - keys   : Feature variables.
                                - values : List of probability masses of each category of the corresponding feature.
        rewards_queries : Dict, - keys   : Feature variables.
                                - values : Reward of querying the value of the corresponding feature.
        encoder         : Object of class Encoder, the encoder mapping states to their one-hot representation.
        r_plus          : Int, reward of a correct report (default=5).
        r_minus         : Int, reward of an incorrect report (default=-5).
        split           : Int, the split point we use to define our concept.
        d               : Int, the number of feature variables.
        data_point      : List of length d, the data point starting the episode.
        label           : Boolean, the true label of data_point.
        state           : Object of class State, the current state.
        done            : Boolean, whether the episode is finished or not.
        """
        
        self.generator = generator
        self.categories = [len(v) for v in self.generator.values()]
        self.d = len(self.categories)
        self.rewards_queries = rewards_queries
        self.encoder = encoder
        self.r_plus = r_plus
        self.r_minus = r_minus
        self.split = split
        
    def generate(self):
        """
        Description
        --------------
        Generate a data point.
        
        Parameters
        --------------
        
        Returns
        --------------
        List with the values of each feature, it represents the data point.
        """
        
        return [np.random.choice(self.categories[i], p=self.generator[i]) for i in range(self.d)]
    
    def concept(self, data_point):
        """
        Description
        --------------
        Define the concept labeling the data points. we can define it as a decision tree for example.
        
        Parameters
        --------------
        data_point : List of length d, the data point to label.
        
        Returns
        --------------
        Int in {0, 1}, the label of the data point.
        """
        
        label = True
        i = 0
        while label and i <= self.d-1:
            if data_point[i] >= self.split:
                label = False
                
            i += 1
            
        return label
    
    def reset(self, data_point=None):
        """
        Description
        --------------
        Reset the environment to start a new episode. If data_point is specified, start the episode from it,
        otherwise generate it.
        
        Parameters
        --------------
        data_point : List of length d, the data point to label (default=None).
        
        Returns
        --------------
        """
        
        self.data_point = self.generate() if data_point is None else data_point
        self.label = self.concept(self.data_point)
        self.state = StateDQN([np.NaN for i in range(self.d)], self.encoder, categories=self.categories)
        self.done = False
    
    def step(self, action):
        """
        Description
        --------------
        Interract with the environment through an action taken at the current state.
        
        Parameters
        --------------
        action : Int in {0, ..., d-1, d, d+1}, 
                 - 0, ..., d-1 represent query actions.
                 - d, d+1 represent report actions.
        
        Returns
        --------------
        reward     : Int, the reward of taking this action at the current state.
        next_state : Object of class State, the next state.
        done       : Boolean, whether the episode is finished or not.
        """
        
        # Treating query actions.
        if action <= self.d-1:
            # If it is an allowed query action.
            if np.isnan(self.state.values[action]):
                reward = self.rewards_queries[action]
                values = self.state.values
                values[action] = self.data_point[action] # Reveal the value of the queried feature in the data point.
                self.state = StateDQN(values, self.encoder, self.categories)
                
            # If this query action is not allowed.
            else:
                print('unallowed')
            
        # Treating report actions.
        else:
            reward = self.r_plus if (action%self.d) == self.label else self.r_minus
            self.done = True
            
        return reward, self.state, self.done







