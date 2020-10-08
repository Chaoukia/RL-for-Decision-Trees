from utils import *


class Environment:
    """
    Description
    --------------
    Asbtract representation of an environment
    """
    
    def __init__(self, rewards_queries, r_plus=5, r_minus=-5, encoder=None):
        """
        Description
        --------------
        Constructor of class Environment.
        
        Parameters & Attributes
        --------------
        rewards_queries : Dict, - keys   : Feature variables.
                                - values : Reward of querying the value of the corresponding feature.
        r_plus          : Int, reward of a correct report (default=5).
        r_minus         : Int, reward of an incorrect report (default=-5).
        encoder         : None or an object of class Encoder. We need to specify an encoder when working with DQN or Actor Critic.
        """
        
        self.rewards_queries = rewards_queries
        self.r_plus = r_plus
        self.r_minus = r_minus
        self.encoder = encoder
        
    def generate(self):
        """
        Description
        --------------
        Generate a data point and its label.
        """
        
        raise NotImplementedError
        
    def reset(self, data_point=None, label=None):
        """
        Description
        --------------
        Reset the environment to start a new episode. If data_point is specified, start the episode from it,
        otherwise generate it.
        
        Parameters
        --------------
        data_point : None or List of length d, the data point to label.
        label      : None or Int, the data point label.
        
        Returns
        --------------
        """
        
        if data_point is None:
            self.data_point, self.label = self.generate()
                        
        else:
            self.data_point, self.label = data_point, label
        
        self.state = State([np.NaN for i in range(self.d)], categories=self.categories, encoder=self.encoder)
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
            self.state = State(values, categories=self.categories, encoder=self.encoder)
            
        # Treating report actions.
        else:
            reward = self.r_plus if (action%self.d) == self.label else self.r_minus
            self.done = True
            
        return reward, self.state, self.done


class EnvironmentSynthetic(Environment):
    """
    Description
    --------------
    Class representing an environment with synthetic data.
    """
    
    def __init__(self, generator, rewards_queries, r_plus=5, r_minus=-5, split=3, encoder=None):
        """
        Description
        --------------
        Constructor of class EnvironmentSynthetic.
        
        Parameters & Attributes
        --------------
        generator       : Dict, - keys   : Feature variables.
                                - values : List of probability masses of each category of the corresponding feature.
        rewards_queries : Dict, - keys   : Feature variables.
                                - values : Reward of querying the value of the corresponding feature.
        r_plus          : Int, reward of a correct report (default=5).
        r_minus         : Int, reward of an incorrect report (default=-5).
        split           : Int, the splitting criterion (see the paper for more details).
        encoder         : None or an object of class Encoder. We need to specify an encoder when working with DQN or Actor Critic.
        categories      : List of length the number of feature variables, categories[i] is the number of classes of feature variable i.
        d               : Int, the input space dimension.
        """
        
        super(EnvironmentSynthetic, self).__init__(rewards_queries, r_plus, r_minus, encoder)
        self.generator = generator
        self.categories = [len(v) for v in self.generator.values()]
        self.d = len(self.categories)
        self.split = split
        
    def generate(self):
        """
        Description
        --------------
        Generate a data point and its label.
        
        Parameters
        --------------
        
        Returns
        --------------
        List with the values of each feature, it represents the data point.
        Int, the label corresponding to the data point.
        """
            
        data_point = [np.random.choice(self.categories[i], p=self.generator[i]) for i in range(self.d)]
        return data_point, self.concept(data_point)
    
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


class EnvironmentReal(Environment):
    """
    Description
    --------------
    Class representing an environment with real data.
    """
    
    def __init__(self, data, rewards_queries, r_plus=5, r_minus=-5, encoder=None):
        """
        Description
        --------------
        Constructor of class EnvironmentReal.
        
        Parameters & Attributes
        --------------
        data            : 2D np.array, the data matrix.
        rewards_queries : Dict, - keys   : Feature variables.
                                - values : Reward of querying the value of the corresponding feature.
        r_plus          : Int, reward of a correct report (default=5).
        r_minus         : Int, reward of an incorrect report (default=-5).
        encoder         : None or an object of class Encoder. We need to specify an encoder when working with DQN or Actor Critic.
        categories      : List of length the number of feature variables, categories[i] is the number of classes of feature variable i.
        d               : Int, the input space dimension.
        maps            : Dict converting categorical variables to numerical variables.
                            - keys   : Int, Feature variable number.
                            - values : Dict mapping the categorical classes to numbers.
        index           : Int, since we are modeling a data stream, index represents the current data instance in the data matrix.
        """
        
        super(EnvironmentReal, self).__init__(rewards_queries, r_plus, r_minus, encoder)
        self.data = data
        self.categories = [len(set(data[:, j])) for j in range(data.shape[1]-1)]
        self.d = len(self.categories)
        self.maps = self.build_maps()
        self.index = 0
        
    def build_maps(self):
        """
        Description
        --------------
        Build the maps dictionnary mapping the categorical classes of each feature to numerical values.
        
        Parameters
        --------------
        
        Returns
        --------------
        """
        
        maps = {}
        for j in range(self.data.shape[1]):
            categories = list(set(self.data[:, j]))
            categories.sort()
            maps_j = {}
            for i, category in enumerate(categories):
                maps_j[category] = i

            maps[j] = maps_j
            
        return maps
    
    def preprocess(self, u):
        """
        Description
        --------------
        Preprocess an instance of the data matrix by converting it to a numerical variable.
        
        Parameters
        --------------
        u : 1D np.array, the instance to preprocess.
        
        Returns
        --------------
        x : 1D np.array, the converted instance.
        y : Int, the corresponding label in the data matrix.
        """
        x = []
        for j, category in enumerate(u[:-1]):
            x.append(self.maps[j][category])

        y = self.maps[len(u)-1][u[-1]]
        return x, y
    
    def generate(self):
        """
        Description
        --------------
        Generate a data point and its label and move the index to the next instance in the data matrix.
        
        Parameters
        --------------
        
        Returns
        --------------
        x : 1D np.array, the converted current instance.
        y : Int, the corresponding label in the data matrix.
        """
        
        x, y = self.preprocess(self.data[self.index, :])
        self.index += 1
        if self.index == self.data.shape[0]:
            self.index = 0
            
        return x, y


class EnvironmentDrift(Environment):
    """
    Description
    --------------
    Class representing an environment with concept drift.
    """
    
    episode = 0
    
    def __init__(self, generator, rewards_queries, r_plus=5, r_minus=-5, split_1=4, split_2=3, encoder=None):
        """
        Description
        --------------
        Constructor of class EnvironmentSynthetic.
        
        Parameters & Attributes
        --------------
        generator       : Dict, - keys   : Feature variables.
                                - values : List of probability masses of each category of the corresponding feature.
        rewards_queries : Dict, - keys   : Feature variables.
                                - values : Reward of querying the value of the corresponding feature.
        r_plus          : Int, reward of a correct report (default=5).
        r_minus         : Int, reward of an incorrect report (default=-5).
        split_1         : Int, the first splitting criterion (see the paper for more details).
        split_2         : Int, the second splitting criterion (see the paper for more details).
        encoder         : None or an object of class Encoder. We need to specify an encoder when working with DQN or Actor Critic.
        categories      : List of length the number of feature variables, categories[i] is the number of classes of feature variable i.
        d               : Int, the input space dimension.
        """
        
        super(EnvironmentDrift, self).__init__(rewards_queries, r_plus, r_minus, encoder)
        
        self.generator = generator
        self.categories = [len(v) for v in self.generator.values()]
        self.d = len(self.categories)
        self.split_1 = split_1
        self.split_2 = split_2
        
    def generate(self):
        """
        Description
        --------------
        Generate a data point and its label
        
        Parameters
        --------------
        
        Returns
        --------------
        data_point : List with the values of each feature representing the current data point.
        label      : Int, the label corresponding to data_point.
        """
        
        data_point = [np.random.choice(self.categories[i], p=self.generator[i]) for i in range(self.d)]
        b = np.random.binomial(1, 1/(1 + np.exp(-(self.episode - 5e4)/2e3)))
        if b == 0:
            label = self.concept_1(data_point)
            
        else:
            label = self.concept_2(data_point)
            
        EnvironmentDrift.episode += 1
        return data_point, label
    
    def concept_1(self, data_point):
        """
        Description
        --------------
        Define the first labeling concept.
        
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
            if data_point[i] >= self.split_1:
                label = False
                
            i += 1
            
        return label
    
    def concept_2(self, data_point):
        """
        Description
        --------------
        Define the second labeling concept.
        
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
            if data_point[i] == self.split_2:
                label = False
                
            i += 1
            
        return label