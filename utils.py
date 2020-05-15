import numpy as np


class State:
    """
    Description
    --------------
    Class representing a state, it also serves as Node representation
    for our Breadth First Search
    """
    
    def __init__(self, values, categories=[4, 3, 3, 3, 2, 4]):
        """
        Description
        --------------
        Constructor of class State.
        
        Parameters & Attributes
        --------------
        values     : List of length d (Input dimension):
                         - values[i] = NaN if i is an unobsorved feature.
                         - values[i] = value of feature i if it is observed.
        categories : List of length d where categories[i] is the number of categories feature i can take.
        observed   : List containing the observed features at this state.
        unobserved : List containing the unobserved features at this state.
        empty      : Boolean, whether it is the empty state or not.
        complete   : Boolean, whether all features are observed or not.
        """
        
        d = len(values)
        values_nans = np.isnan(values)
        self.values = values
        self.categories = categories
        self.observed = np.arange(d)[np.invert(values_nans)]
        self.unobserved = np.arange(d)[values_nans]  # These are also the allowed query actions at this state
        self.empty = (len(self.observed) == 0)
        self.complete = (len(self.unobserved) == 0)
        
    def __repr__(self):
        """
        Description
        --------------
        String representation of the state.
        """
        
        s = '| '
        for x in self.values:
            s += str(x) + ' | '
            
        return s
    
    def __str__(self):
        """
        Description
        --------------
        String representation of the state.
        """
        
        s = '| '
        for x in self.values:
            s += str(x) + ' | '
            
        return s
    
    def children(self):
        """
        Description
        --------------
        Provide the all possible states that we can transit to from the current state
        by observing one unobserved feature
        
        Parameters
        --------------
        
        Returns
        --------------
        children : Set of the children states
        """
        
        children = set([])
        # Loop over the unobserved features.
        for j in self.unobserved:
            # Loop over the possible categories of all unobserved feature j for the empty state.
            if self.empty:
                for k in range(self.categories[j]):
                    values = self.values.copy()
                    values[j] = k
                    child = State(values, self.categories)
                    children.add(child)
                    
            else:
                # Check if j comes after the first observed feature to avoid treating the same nodes in Breadth-First-Search.
                if j > self.observed[-1]:
                    for k in range(self.categories[j]):
                        values = self.values.copy()
                        values[j] = k
                        child = State(values, self.categories)
                        children.add(child)
                
        return children

    
class Encoder:
    """
    Description
    --------------
    Class represeting the one-hot encoder of the states.
    """
    
    def __init__(self, categories=[4, 3, 3, 3, 2, 4]):
        """
        Description
        --------------
        Constructor of class Encoder.
        
        Parameters & Attributes
        --------------
        categories : List of length d where categories[i] is the number of categories feature i can take.
        d          : Int, the input dimension.
        dim        : Int, the one-hot encoded representation dimension
        
        """
        
        self.categories = categories
        self.d = len(categories)
        self.dim = np.sum(categories) + self.d
        
    def transform(self, state_values):
        """
        Description
        --------------
        Encode the vector state representation with dummies.
        
        Parameters & Attributes
        --------------
        state_values : List of length d where the ith entry is either NaN or the the feature value.
        
        Returns
        --------------
        state_one_hot : 2D np.array representing the one-hot encoded state.
        """
    
        s = 0
        state_one_hot = np.zeros((1, self.dim), dtype = np.float32)
        for i, value in enumerate(state_values):
            if np.isnan(value):
                state_one_hot[0, self.categories[i] + s] = 1

            else:
                state_one_hot[0, value + s] = 1

            s += self.categories[i]+1

        return state_one_hot
    
    
class StateDQN:
    """
    Description
    --------------
    Class representing a state, it also serves as Node representation
    for our Breadth First Search
    """
    
    def __init__(self, values, encoder, categories=[4, 3, 3, 3, 2, 4]):
        """
        Description
        --------------
        Constructor of class State.
        
        Parameters & Attributes
        --------------
        values     : List of length d (Input dimension):
                         - values[i] = NaN if i is an unobsorved feature.
                         - values[i] = value of feature i if it is observed.
        encocer    : sklearn.preprocessing._encoders.OneHotEncoder object.
        categories : List of length d where categories[i] is the number of categories feature i can take.
        observed   : List containing the observed features at this state.
        unobserved : List containing the unobserved features at this state.
        empty      : Boolean, whether it is the empty state or not.
        complete   : Boolean, whether all features are observed or not.
        """
        
        d = len(values)
        values_nans = np.isnan(values)
        self.encoder = encoder  # One-hot encoder, used in the approximate RL framework for state representation.
        self.values = values
        self.values_encoded = self.encode()
        self.categories = categories
        self.observed = np.arange(d)[np.invert(values_nans)]
        self.unobserved = np.arange(d)[values_nans]  # These are also the allowed query actions at this state
        self.empty = (len(self.observed) == 0)
        self.complete = (len(self.unobserved) == 0)
        
    def encode(self):
        """
        Description
        --------------
        Encode the state with dummy variables. To be used when a one-hot encoder is defined.
        
        Parameters
        --------------
        
        Returns
        --------------
        np.array of shape (1, #one_hot_representation_dim)
        """
        
        return self.encoder.transform(self.values)
        
    def __repr__(self):
        """
        Description
        --------------
        String representation of the state.
        """
        
        s = '| '
        for x in self.values:
            s += str(x) + ' | '
            
        return s
    
    def __str__(self):
        """
        Description
        --------------
        String representation of the state.
        """
        
        s = '| '
        for x in self.values:
            s += str(x) + ' | '
            
        return s
    

























