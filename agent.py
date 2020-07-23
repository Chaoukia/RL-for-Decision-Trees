from utils import *
from models import *
from memory import *
from queue import Queue
from copy import deepcopy
from nltk import Tree
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import pickle as pkl
import os


class Agent:
    """
    Description
    --------------
    Class describing an RL agent.
    """
    
    def __init__(self, gamma=.9, categories=[4, 3, 3, 3, 2, 4], labels=[0, 1], init=True):
        """
        Description
        --------------
        Constructor of class Agent;
        
        Parameters & Attributes
        --------------
        gamma           : Float in ]0, 1[, the discount factor (default=0.9).
        categories      : List of length d where categories[i] is the number of categories feature i can take.
        labels          : List of the possible labels.
        Q               : Dict, - keys   : String representations of the states.
                                - values : List of length d+2 representing the state-action values.
        PI              : Dict, - keys   : String representations of the states.
                                - values : List of length d+2 representing the exploratry policy.
        V               : Dict, - keys   : String representations of the states.
                           - values : List of length d+2 representing the number of visits of state-action pairs.
        policy          : Dict, - keys   : String representations of the states.
                                - values : Int representing the action to take at that state.
        d               : Int, the number of feature variables.
        b               : Int, the number of class labels.
        actions         : List of all actions.
        actions_report  : List of report actions.
        """
        
        self.gamma = gamma
        self.d = len(categories)
        self.b = len(labels)
        self.categories = categories
        self.labels = labels
        self.actions = range(self.d + len(labels))
        self.actions_report = [self.d + label for label in labels]
        if init:
            self.Q, self.PI, self.V = self.initialize()
            
        else:
            self.Q, self.PI, self.V = {}, {}, {}
            
        self.policy = None
        
    def initialize(self):
        """
        Description
        --------------
        Initialize the dictionaries Q, PI and V.
        
        Parameters
        --------------
        
        Returns
        --------------
        Q  : Dict, - keys   : String representations of the states.
                   - values : List of length d+2 representing the state-action values.
        PI : Dict, - keys   : String representations of the states.
                   - values : List of length d+2 representing the exploratry policy.
        V  : Dict, - keys   : String representations of the states.
                   - values : List of length d+2 representing the number of visits of state-action pairs.
        """
        
        # Initialize to empty dictionaries.
        Q, PI, V = {}, {}, {}
        
        # The root is the empty state.
        values_root = [np.NaN for i in range(self.d)]
        root = State(values_root, self.categories)
        root_repr = repr(root)  # String representation of the root.
        
        n_queries = len(root.unobserved)
        Q[root_repr], PI[root_repr], V[root_repr] = np.full(self.d + self.b, np.NaN), np.zeros(self.d + self.b), np.full(self.d + self.b, np.NaN)
        for query in root.unobserved:
            Q[root_repr][query], PI[root_repr][query], V[root_repr][query] = 0, 1/n_queries, 0
    
        # Initialize the FIFO queue and the put the root children in it.
        queue = Queue()
        for child in root.children():
            queue.put(child)

        # Breadth-First-Search
        while not queue.empty():
            node = queue.get()
            node_repr = repr(node)
            n_queries = len(node.unobserved)
            Q[node_repr], PI[node_repr], V[node_repr] = np.full(self.d + self.b, np.NaN), np.zeros(self.d + self.b), np.full(self.d + self.b, np.NaN)
            # Loop over the allowed query actions at the state described by this node.
            for query in node.unobserved:
                Q[node_repr][query], PI[node_repr][query], V[node_repr][query] = 0, 1/n_queries, 0
                
            # Loop over the report actions at the state described by this node.
            for report in self.actions_report:
                Q[node_repr][report], V[node_repr][report] = 0, 0

            for child in node.children():
                queue.put(child)
                            
        return Q, PI, V
    
    def action_explore(self, state):
        """
        Description
        --------------
        Choose an action at a state according to the exploratory policy and update dictionaries PI and V.
        
        Parameters
        --------------
        state  : Object of class State.
        
        Returns
        --------------
        action : Int in {0, ..., d-1, d, d+1}.
        """
        
        state_repr = repr(state)
        action = np.random.choice(self.actions, p=self.PI[state_repr])
        self.PI[state_repr][action] = 1/(1/self.PI[state_repr][action] + 1) # Update the exploration probability for this action
        self.PI[state_repr] = self.PI[state_repr]/self.PI[state_repr].sum() # Renormalize the probabilities.
        self.V[state_repr][action] += 1  # Update the number of visits to this state-action pair.
        return int(action)
    
    def action(self, state):
        """
        Description
        --------------
        Choose an action at a state according to the greedy policy (need to run greedy_policy() method before).
        
        Parameters
        --------------
        state  : Object of class State.
        
        Returns
        --------------
        action : Int in {0, ..., d-1, d, d+1}.
        """
        
        return self.policy[repr(state)]
    
    def greedy_policy(self):
        """
        Description
        --------------
        Compute the greedy policy w.r.t the current dictionnary Q of Q-values.
        
        Parameters
        --------------
        """
        
        self.policy = {}
        for state_repr in self.Q:
            self.policy[state_repr] = np.nanargmax(self.Q[state_repr])
            
    def q_learning(self, env, n_episodes=100, n_save=1000, path_save='q_learning/'):
        """
        Description
        --------------
        Run Q-learning algorithm to update the Q dictionary and converge to the optimal one.
        
        Parameters
        --------------
        env        : Object of class Environment.
        n_episodes : Int, the number of episodes.
        
        Returns
        --------------
        """
        
        if not os.path.isdir(path_save):
            os.mkdir(path_save)
        
        env.reset()
        state = deepcopy(env.state)
        state_repr = repr(state)
        done = env.done
        for i in range(n_episodes):
            while not state.complete:
                # Reports are not allowed at the empty state.
                if not state.empty:
                    # Explore report actions for non-empty states.
                    for action_report in self.actions_report:
                        self.V[state_repr][action_report] += 1  # Update the #Visits
                        reward = env.r_plus if (action_report%self.d) == env.label else env.r_minus  # Calculate the reward.
                        td = reward - self.Q[state_repr][action_report]   # Calculate the temporal difference
                        # Update the Q-value.
                        self.Q[state_repr][action_report] = self.Q[state_repr][action_report] + td/self.V[state_repr][action_report]
                    
                # Take action according to the exploratory policy (only query actions)
                action_query = self.action_explore(state)
                # Get the corresponding transition from the environment.
                reward, next_state, done = env.step(action_query)
                next_state_repr = repr(next_state)
                # Compute the temporal difference.
                td = reward + self.gamma*np.nanmax(self.Q[next_state_repr]) - self.Q[state_repr][action_query]
                # Update the Q-value.
                self.Q[state_repr][action_query] = self.Q[state_repr][action_query] + td/self.V[state_repr][action_query]
                state = deepcopy(next_state)
                state_repr = repr(state)
                
            # Explore report actions for complete states.
            for action_report in self.actions_report:
                self.V[state_repr][action_report] += 1
                reward = env.r_plus if (action_report%self.d) == env.label else env.r_minus
                td = reward - self.Q[state_repr][action_report]
                self.Q[state_repr][action_report] = self.Q[state_repr][action_report] + td/self.V[state_repr][action_report]
                
            env.reset()
            state = deepcopy(env.state)
            state_repr = repr(state)
            done = env.done
            
            if i%n_save == 0:
                print('Episode : %d' %i)
                self.save_weights(path_save + 'q_learning_weights_' + str(i) + '.pkl')
            
    def predict(self, env, data_point):
        """
        Description
        --------------
        Predict the label of a given data point.
        
        Parameters
        --------------
        env        : Object of class Environment.
        data_point : List of length d, the data point to label.
        
        Returns
        --------------
        """
        
        env.reset(data_point)
        state = deepcopy(env.state)
        done = env.done
        while not done:
            # Take action according to the exploratory policy.
            action = self.action(state)
            # Get the corresponding transition from the environment.
            reward, next_state, done = env.step(action)
            state = deepcopy(next_state)
            
        return action%self.d
    
    def test(self, env, n_test=1000):
        """
        Description
        --------------
        Test the agent on n_test data points generated by env.
        
        Parameters
        --------------
        env      : Object of class Environment.
        n_test   : Int, number of data points to test the agent on.
        
        Returns
        --------------
        accuracy : FLoat in [0, 1], the accuracy of the agent on this test.
        """
        
        valids = 0
        for i in range(n_test):
            data_point = env.generate()
            env.reset(data_point)
            label_pred, label_true = self.predict(env, data_point), env.label
            valids += (label_pred==label_true)
            
        return valids/n_test
    
    def save_weights(self, path):
        """
        Description
        --------------
        Save the agents q-dictionary.
        
        Parameters
        --------------
        path: String, path to a .pkl file containing q-dictionary.
        
        Returns
        --------------
        """
        
        with open(path, 'wb') as f:
            pkl.dump(self.Q, f)
    
    def load_weights(self, path):
        """
        Description
        --------------
        Load the q-dictionary.
        
        Parameters
        --------------
        path: String, path to a .pkl file containing q-dictionary.
        
        Returns
        --------------
        """
        
        with open(path, 'rb') as f:
            self.Q = pkl.load(f)
    
    def children(self, state):
        """
        Description
        --------------
        Give the possible outcomes of taking the greedy policy at the considered state.
        
        Parameters
        --------------
        state : Object of class State.
        
        Returns
        --------------
        children : Set of objects of class State.
        action   : Int, action taken at state with the agent policy.
        """
        
        children = []
        action = self.action(state)
        if action >= self.d: return children, action
        for category in range(self.categories[action]):
            values = state.values.copy()
            values[action] = category
            children.append(State(values, self.categories))

        return children, action
    
    def build_string_state(self, state):
        """
        Description
        --------------
        Build string representation of the agent decision (with parentheses) starting from state.
        
        Parameters
        --------------
        state : Object of class State.
        
        Returns
        --------------
        string : String representation of a tree.
        """
        
        l, action = self.children(state)
        if action >= self.d: return str(self.action(state)%self.d) + ''
        string = ''
        for child in l:
            string += '(X_' + str(action) + '=' + str(child.values[action]) + ' ' + self.build_string_state(child) + ') '

        return string
    
    def build_string(self):
        """
        Description
        --------------
        Build string representation of the agent decision.
        
        Parameters
        --------------
        
        Returns
        --------------
        string : String representation of a tree.
        """
        
        return '( ' + self.build_string_state(State([np.NaN for i in range(self.d)], self.categories)) + ')'
    
    def plot_tree(self):
        """
        Description
        --------------
        Plot the agent's decision tree.
        
        Parameters
        --------------
        
        Returns
        --------------
        nltk tree object, helpful to visualize the agent's decision tree policy.
        """

        return Tree.fromstring(self.build_string())
    
    
    
    
class AgentDQN:
    """
    Description
    --------------
    Class describing a DQN agent
    """
    
    def __init__(self, gamma=.9, categories=[4, 3, 3, 3, 2, 4], labels=[0, 1], max_size_queries=int(1e4), max_size_reports=int(1e3)):
        """
        Description
        --------------
        Constructor of class AgentDQN.
        
        Parameters & Attributes
        --------------
        gamma            : Float in ]0, 1[, the discount factor (default=0.9).
        categories       : List of length d where categories[i] is the number of categories feature i can take.
        labels           : List of the possible labels.
        max_size         : Int, the maximum size of the experience replay memory.
        d                : Int, the number of feature variables.
        b                : Int, the number of class labels.
        actions          : List of all actions.
        actions_queries  : List of query actions.
        actions_report   : List of report actions.
        memory_queries   : Object of class Memory, replay buffer containing experiences with query actions only.
        memory_reports   : Object of class Memory, replay buffer containing experiences with report actions only.
        q_network        : Object of class DQNetwork, the current q-network estimating the Q-function.
        q_network_target : Object of class DQNetwork, the target q-network, it should have the same architecture as q_network.

        Returns
        --------------
        """
        
        self.gamma = gamma
        self.categories = categories
        self.labels = labels
        self.max_size_queries = max_size_queries
        self.max_size_reports = max_size_reports
        self.d = len(categories)
        self.b = len(labels)
        self.actions = range(self.d + len(labels))
        self.actions_queries = range(self.d)
        self.actions_report = [self.d + label for label in labels]
        self.memory_queries = Memory(max_size_queries)
        self.memory_reports = Memory(max_size_reports)
        self.q_network = DQNetwork(input_size=np.sum(categories)+self.d, out=self.d+self.b)
        self.q_network_target = DQNetwork(input_size=np.sum(categories)+self.d, out=self.d+self.b)
        
    def action_explore(self, state, epsilon=.1):
        """
        Description
        --------------
        Choose a query action and a report action at a state according to the epsilon-greedy policy.
        
        Parameters
        --------------
        state   : Object of class State.
        epsilon : Float in [0, 1], the probability of taking an action according to the greedy policy.
        
        Returns
        --------------
        query  : Int in {0, ..., d-1}
        report : Int in {d, d+1}
        """
        
        bern = np.random.binomial(1, epsilon)
        
        # Use conditions to avoid unnecessary forward passes on the Q-network.
        if bern == 1:
            query = np.random.choice(state.unobserved)
            report = np.random.randint(self.d, self.d + self.b)
            
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.from_numpy(state.values_encoded))
                # Consider only q-values of allowed queries (for queries)
                q_queries_allowed = q_values[0, state.unobserved]
                # map argmax to the corresponding index in the allowed queries.
                query = state.unobserved[torch.argmax(q_queries_allowed).item()]
                # Take the greedy report action.
                report = torch.argmax(q_values[0, self.d:]).item()+self.d
            
        return query, report
    
    def action(self, state):
        """
        Description
        --------------
        Choose an action at a state greedy policy.
        
        Parameters
        --------------
        state   : Object of class State.
        
        Returns
        --------------
        action : Int in {0, ..., d-1, d, d+1}.
        """
        
        with torch.no_grad():
            q_values = self.q_network(torch.from_numpy(state.values_encoded))
            actions_allowed = list(state.unobserved) + [self.d+i for i in range(self.b)]
            q_values_allowed = q_values[0, actions_allowed]
            return actions_allowed[torch.argmax(q_values_allowed).item()]
    
    def update_target(self):
        """
        Description
        -------------
        Update the parameters of target_model with those of current_model

        Parameters
        -------------
        current_model, target_model : torch models
        """
        
        self.q_network_target.load_state_dict(self.q_network.state_dict())
    
    def pretrain(self, env, n_episodes):
        """
        Description
        --------------
        Fill the memory buffers with experiences in a pretraining phase.
        
        Parameters
        --------------
        env        : Object of class EnvironmentDQN.
        n_episodes : Int, number of episodes.
        
        Returns
        --------------
        """
        
        for episode in range(n_episodes):
            env.reset()
            state = deepcopy(env.state)
            while not state.complete:
                query, report = np.random.choice(state.unobserved), np.random.randint(self.d, self.d + self.b)
                reward, next_state, _ = env.step(report)
                self.memory_reports.add(state, report, reward, next_state)
                reward, next_state, _ = env.step(query)
                self.memory_queries.add(state, query, reward, next_state)
                state = deepcopy(next_state)
                
            report = np.random.randint(self.d, self.d + self.b)
            reward, next_state, _ = env.step(report)
            self.memory_reports.add(state, report, reward, next_state)   
            
    def train(self, env, n_train=1000, n_pretrain=100, epsilon_start=1, epsilon_stop=1e-4, decay_rate=1e-3, n_learn=5, batch_size=32, lr=1e-4, log_dir='runs_dqn/', 
             n_write=10, max_tau=50, double=True, n_save=1000, path_save='dqn_weights/'):
        """
        Description
        --------------
        Explore the environment and train the agent.
        
        Parameters
        --------------
        env           : Object of class EnvironmentDQN.
        n_train       : Int, number of training episodes.
        n_pretrain    : Int, number of pretraining episode.
        epsilon_start : Float in [0, 1], the starting epsilon parameter of the epsilon-greedy policy, it should be high.
        epsilon_stop  : Float in [0, 1], the last epsilon parameter of the epsilon-greedy policy, it should be small.
        decay_rate    : Float in [0, 1], the decay rate of epsilon.
        n_learn       : Int, the number of iterations between two consecutive learning phases.
        batch_size    : Int, the batch size for both replay buffers.
        lr            : Float, the learning rate.
        log_dir       : String, path of the folder where tensorboard events are stored.
        n_write       : Int, the number of iterations between two consecutive events writings.
        max_tau       : Int, the number of iterations between two consecutive target network updates.
        double        : Boolean, whether to use Double DQN or just DQN.
        n_save        : Int, the number of episodes between two consecutive saved models.
        path_save     : String, the path of the folder where weights will be saved.
        
        Returns
        --------------
        """
        
        writer = SummaryWriter(log_dir=log_dir)
        if not os.path.isdir(path_save):
            os.mkdir(path_save)
        
        # Pretrain the agent.
        self.pretrain(env, n_pretrain)
        epsilon = epsilon_start
        it = 0
        optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        variables = {'losses' : []}
        tau = 0
        for episode in range(n_train):
            env.reset()
            state = deepcopy(env.state)
            while not state.complete:
                query, report = self.action_explore(state, epsilon)
                it += 1
                tau += 1
                epsilon = epsilon_stop + (epsilon_start - epsilon_stop)*np.exp(-decay_rate*it)
                reward, next_state, _ = env.step(report)
                self.memory_reports.add(state, report, reward, next_state)
                reward, next_state, _ = env.step(query)
                self.memory_queries.add(state, query, reward, next_state)
                state = deepcopy(next_state)
                
                # Learning phase
                if it%n_learn == 0:
                    batch_queries = self.memory_queries.sample(batch_size)
                    batch_reports = self.memory_reports.sample(batch_size)
                    
                    states_batch_queries, states_batch_reports = torch.cat(batch_queries.state), torch.cat(batch_reports.state)
                    actions_batch_queries, actions_batch_reports = np.concatenate(batch_queries.action), np.concatenate(batch_reports.action)
                    rewards_batch_queries, rewards_batch_reports = torch.cat(batch_queries.reward), torch.cat(batch_reports.reward)
                    next_states_batch_queries, next_states_batch_reports = torch.cat(batch_queries.next_state), torch.cat(batch_reports.next_state)
                    next_states_batch_unallowed = torch.cat(batch_queries.next_state_unallowed)
                    
                    with torch.no_grad():
                        if double:
                            q_values = self.q_network(next_states_batch_queries)
                            q_values = q_values - 100*next_states_batch_unallowed
                            actions_values = torch.argmax(q_values, dim=1)
                            q_targets_queries = rewards_batch_queries + self.gamma*self.q_network_target(next_states_batch_queries)[np.arange(batch_size), actions_values].detach()

                        else:
                            q_values = self.q_network_target(next_states_batch_queries)
                            q_values = q_values - 100*next_states_batch_unallowed
                            q_targets_queries = rewards_batch_queries + self.gamma*torch.max(q_values, dim=1).values

                    q_values_queries = self.q_network(states_batch_queries)[np.arange(batch_size), actions_batch_queries]
                    q_targets_reports = rewards_batch_reports
                    q_values_reports = self.q_network(states_batch_reports)[np.arange(batch_size), actions_batch_reports]
                    
                    q_targets = torch.cat((q_targets_queries, q_targets_reports))
                    q_values = torch.cat((q_values_queries, q_values_reports))
                    
                    loss = F.mse_loss(q_values, q_targets)
                    variables['losses'].append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    if it%n_write == 0:
                        writer.add_scalar('Loss', np.mean(variables['losses']), it)
                        
                if tau == max_tau:
                    self.update_target()
                    tau = 0
                    
            # The last action of an episode should be a report, take it according to the epsilon-greedy policy.
            bern = np.random.binomial(1, epsilon)
            if bern == 1:
                report = np.random.randint(self.d, self.d + self.b)

            else:
                with torch.no_grad():
                    q_values = self.q_network(torch.from_numpy(state.values_encoded))
                    report = torch.argmax(q_values[0, self.d:]).item()+self.d
                        
            epsilon = epsilon_stop + (epsilon_start - epsilon_stop)*np.exp(-decay_rate*it)
            reward, next_state, _ = env.step(report)
            self.memory_reports.add(state, report, reward, next_state)
            
            if episode%n_save == 0:
                print('Episode : %d, epsilon : %.3f' %(episode, epsilon))
                self.save_weights(path_save + 'dqn_weights_' + str(episode) + '.pth')
                        
        writer.close()
        
    def predict(self, env, data_point):
        """
        Description
        --------------
        Predict the label of a data point.
        
        Parameters
        --------------
        env        : Object of class EnvironmentDQN.
        data_point : List of length d, the data point to label.
        
        Returns
        --------------
        Int in {0, 1}, the predicted label.
        """
        
        env.reset(data_point)
        state = env.state
        while not env.done:
            action = self.action(state)
            env.step(action)
            state = env.state
        
        return action%self.d
        
    def test(self, env, n_test=1000):
        """
        Description
        --------------
        Test the agent on n_test data points generated by env.
        
        Parameters
        --------------
        env      : Object of class EnvironmentDQN.
        n_test   : Int, number of data points to test the agent on.
        
        Returns
        --------------
        accuracy : FLoat in [0, 1], the accuracy of the agent on this test.
        """
        
        valids = 0
        for i in range(n_test):
            data_point = env.generate()
            env.reset(data_point)
            label_pred, label_true = self.predict(env, data_point), env.label
            valids += (label_pred==label_true)
            
        return valids/n_test
    
    def save_weights(self, path):
        """
        Description
        --------------
        Save the agents q-network weights.
        
        Parameters
        --------------
        path: String, path to a .pth file containing the weights of a q-network.
        
        Returns
        --------------
        """
        
        torch.save(self.q_network.state_dict(), path)
    
    def load_weights(self, path):
        """
        Description
        --------------
        Load the weights of a q-network.
        
        Parameters
        --------------
        path: String, path to a .pth file containing the weights of a q-network.
        
        Returns
        --------------
        """
        
        self.q_network.load_state_dict(torch.load(path))
        
    def children(self, state):
        """
        Description
        --------------
        Give the possible outcomes of taking the greedy policy at the considered state.
        
        Parameters
        --------------
        state : Object of class StateDQN.
        
        Returns
        --------------
        children : Set of objects of class State.
        action   : Int, action taken at state with the agent policy.
        """
        
        children = []
        action = self.action(state)
        if action >= self.d: return children, action
        for category in range(self.categories[action]):
            values = state.values.copy()
            values[action] = category
            children.append(StateDQN(values, state.encoder, self.categories))

        return children, action
    
    def build_string_state(self, state):
        """
        Description
        --------------
        Build string representation of the agent decision (with parentheses) starting from state.
        
        Parameters
        --------------
        state : Object of class StateDQN.
        
        Returns
        --------------
        string : String representation of a tree.
        """
        
        l, action = self.children(state)
        if action >= self.d: return str(self.action(state)%self.d) + ''
        string = ''
        for child in l:
            string += '(X_' + str(action) + '=' + str(child.values[action]) + ' ' + self.build_string_state(child) + ') '

        return string
    
    def build_string(self, encoder):
        """
        Description
        --------------
        Build string representation of the agent decision.
        
        Parameters
        --------------
        encoder : Object of class Encoder, the encoder mapping states to their one-hot representation.
        
        Returns
        --------------
        string : String representation of a tree.
        """
        
        return '( ' + self.build_string_state(StateDQN([np.NaN for i in range(self.d)], encoder, self.categories)) + ')'
    
    def plot_tree(self, encoder):
        """
        Description
        --------------
        Plot the agent's decision tree.
        
        Parameters
        --------------
        encoder : Object of class Encoder, the encoder mapping states to their one-hot representation.
        
        Returns
        --------------
        nltk tree object, helpful to visualize the agent's decision tree policy.
        """

        return Tree.fromstring(self.build_string(encoder))
    
    
    
class AgentActorCritic:
    """
    Description
    --------------
    Class describing a DQN agent
    """
    
    def __init__(self, gamma=.9, categories=[4, 3, 3, 3, 2, 4], labels=[0, 1], min_queries=4):
        """
        Description
        --------------
        Constructor of class AgentDQN.
        
        Parameters & Attributes
        --------------
        gamma           : Float in ]0, 1[, the discount factor (default=0.9).
        categories      : List of length d where categories[i] is the number of categories feature i can take.
        labels          : List of the possible labels.
        min_queries     : Int, the minimum number of queries the agent has to perform before being allowed to report a label.
        d               : Int, the number of feature variables.
        b               : Int, the number of class labels.
        actions         : List of all actions.
        actions_queries : List of query actions.
        actions_report  : List of report actions.

        Returns
        --------------
        """
        
        self.gamma = gamma
        self.categories = categories
        self.labels = labels
        self.min_queries = min_queries
        self.d = len(categories)
        self.b = len(labels)
        self.actions = range(self.d + len(labels))
        self.actions_queries = range(self.d)
        self.actions_report = [self.d + label for label in labels]
        self.actor_critic = ActorCritic(input_size=np.sum(categories)+self.d, out=self.d+self.b)
        
    def actions_probas(self, state):
        """
        Description
        --------------
        Calculate the probabilities of the allowed actions at a state.
        
        Parameters
        --------------
        state   : Object of class State.
        
        Returns
        --------------
        actions_probas  : torch.tensor of size #Allowed_actions, with the allowed actions probabilities.
        actions_allowed : List of the allowed actions.
        value           : 1D torch.tensor, the state value estimated by the value head of the Actor-Critic network.
        """
        
        value, actions_output = self.actor_critic(torch.from_numpy(state.values_encoded))
        if len(state.observed) < self.min_queries:
            actions_allowed = list(state.unobserved)
            
        else:
            actions_allowed = list(state.unobserved) + [self.d+i for i in range(self.b)]
            
        actions_probas = F.softmax(actions_output[0, actions_allowed], dim=0)
        return actions_probas, actions_allowed, value
        
    def action(self, state):
        """
        Description
        --------------
        Choose an action at a state by sampling from the current stochastic policy.
        
        Parameters
        --------------
        state  : Object of class State.
        
        Returns
        --------------
        action           : Int in {0, ..., d-1, d, d+1}, action sampled from the stochastic policy of the Actor.
        action_log_proba : Float, the log probability correspoding to the performed action.
        actions_probas   : List of length #Allowed_actions with the estimated probabilities of the allowed actions.
        value            : 1D torch.tensor, the state value estimated by the value head of the Actor-Critic network.
        """
        
        actions_probas, actions_allowed, value = self.actions_probas(state)
        m = Categorical(actions_probas)
        index = m.sample()
        action, action_log_prob = actions_allowed[index], m.log_prob(index)
        return action, action_log_prob, actions_probas, value
    
    def action_greedy(self, state):
        """
        Description
        --------------
        Choose the action maximizing the stochastic policy at the state.
        
        Parameters
        --------------
        state  : Object of class State.
        
        Returns
        --------------
        action : Int in {0, ..., d-1, d, d+1}
        """
        
        with torch.no_grad():
            actions_probas, actions_allowed, _ = self.actions_probas(state)
            return actions_allowed[torch.argmax(actions_probas).item()]
        
        
    def train(self, env, n_train=1000, lr=3e-4, log_dir='runs_actor_critic/', n_save=1000, path_save='actor_critic_weights/', max_step=8, lambd=1e-3, clip_grad=.1):
        """
        Description
        --------------
        Explore the environment and train the agent.
        
        Parameters
        --------------
        env       : Object of class Environment.
        n_train   : Int, number of training episodes.
        lr        : Float, the learning rate.
        log_dir   : String, path of the folder where tensorboard events are stored.
        n_save    : Int, the number of episodes between two consecutive saved models.
        path_save : String, path of the directory where the models weights will be stored.
        max_step  : Int, the number of steps in an episode between two consecutive parameters update.
        lambd     : Float, the entropy loss parameter.
        clip_grad : Float in [0, 1] clipping the norm of the gradient to avoid over-optimistic updates.
        
        Returns
        --------------
        """
        
        writer = SummaryWriter(log_dir=log_dir)
        if not os.path.isdir(path_save):
            os.mkdir(path_save)
        
        optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        it = 0
        for episode in range(n_train):
            env.reset()
            state = deepcopy(env.state)
            episode_len = 0
            episode_rewards = 0
            while not env.done:
                rewards, values, log_probs = [], [], []
                step = 0
                entropy = 0
                while (not env.done) and (step <= max_step):
                    action, action_log_prob, actions_probas, value = self.action(state)
                    reward, next_state, _ = env.step(action)
                    episode_rewards += reward
                    rewards.append(reward)
                    values.append(value)
                    log_probs.append(action_log_prob.reshape(1, 1))
                    state = deepcopy(next_state)
                    entropy += -(actions_probas*torch.log(actions_probas)).sum()
                    step += 1
                    it += 1
                    episode_len += 1

                R = torch.tensor(0, dtype=torch.float32) if env.done else self.actor_critic(torch.from_numpy(state.values_encoded))[0].detach()
                size = len(values)
                values_target = [0 for i in range(size)]
                for t in range(size-1, -1, -1):
                    R = torch.tensor(rewards[t], dtype=torch.float32) + self.gamma*R
                    values_target[t] = R.reshape(1, 1)

                values_target = torch.cat(values_target)
                values_current = torch.cat(values)
                log_probs = torch.cat(log_probs)

                advantage = values_target - values_current
                critic_loss = 0.5*(advantage**2).mean()
                actor_loss = -(log_probs*advantage.detach()).mean()
                entropy_loss = -entropy/size
                loss = actor_loss + critic_loss + lambd*entropy_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), clip_grad)
                optimizer.step()

                writer.add_scalar('Losses/Actor', actor_loss.item(), it)
                writer.add_scalar('Losses/Critic', critic_loss.item(), it)
                writer.add_scalar('Losses/Entropy', entropy_loss, it)
                writer.add_scalar('Losses/Loss', loss.item(), it)
                
            writer.add_scalar('Episode/Return', episode_rewards, episode)
            writer.add_scalar('Episode/Length', episode_len, episode)
            if episode%n_save == 0:
                print('Episode : %d' %(episode))
                self.save_weights(path_save + 'actor_critic_weights_' + str(episode) + '.pth')
                        
        writer.close()
        
    def predict(self, env, data_point):
        """
        Description
        --------------
        Predict the label of a data point.
        
        Parameters
        --------------
        
        Returns
        --------------
        """
        
        env.reset(data_point)
        state = env.state
        while not env.done:
            action = self.action_greedy(state)
            env.step(action)
            state = env.state
        
        return action%self.d
        
    def test(self, env, n_test=1000):
        """
        Description
        --------------
        Test the agent on n_test data points generated by env.
        
        Parameters
        --------------
        env      : Object of class Environment.
        n_test   : Int, number of data points to test the agent on.
        
        Returns
        --------------
        accuracy : FLoat in [0, 1], the accuracy of the agent on this test.
        """
        
        valids = 0
        for i in range(n_test):
            data_point = env.generate()
            env.reset(data_point)
            label_pred, label_true = self.predict(env, data_point), env.label
            valids += (label_pred==label_true)
            
        return valids/n_test
    
    def save_weights(self, path):
        """
        Description
        --------------
        Save the agents q-network weights.
        
        Parameters
        --------------
        path: String, path to a .pth file containing the weights of a q-network.
        
        Returns
        --------------
        """
        
        torch.save(self.actor_critic.state_dict(), path)
    
    def load_weights(self, path):
        """
        Description
        --------------
        Load the weights of a q-network.
        
        Parameters
        --------------
        path: String, path to a .pth file containing the weights of a q-network.
        
        Returns
        --------------
        """
        
        self.actor_critic.load_state_dict(torch.load(path))
        
    def children(self, state):
        """
        Description
        --------------
        Give the possible outcomes of taking the greedy policy at the considered state.
        
        Parameters
        --------------
        state : Object of class State.
        
        Returns
        --------------
        children : Set of objects of class State.
        action   : Int, action taken at state with the agent policy.
        """
        
        children = []
        action = self.action_greedy(state)
        if action >= self.d: return children, action
        for category in range(self.categories[action]):
            values = state.values.copy()
            values[action] = category
            children.append(StateDQN(values, state.encoder, self.categories))

        return children, action
    
    def build_string_state(self, state):
        """
        Description
        --------------
        Build string representation of the agent decision (with parentheses) starting from state.
        
        Parameters
        --------------
        state : Object of class State.
        
        Returns
        --------------
        string : String representation of a tree.
        """
        
        l, action = self.children(state)
        if action >= self.d: return str(self.action_greedy(state)%self.d) + ''
        string = ''
        for child in l:
            string += '(X_' + str(action) + '=' + str(child.values[action]) + ' ' + self.build_string_state(child) + ') '

        return string
    
    def build_string(self, encoder):
        """
        Description
        --------------
        Build string representation of the agent decision.
        
        Parameters
        --------------
        
        Returns
        --------------
        string : String representation of a tree.
        """
        
        return '( ' + self.build_string_state(StateDQN([np.NaN for i in range(self.d)], encoder, self.categories)) + ')'
    
    def plot_tree(self, encoder):
        """
        Description
        --------------
        Plot the agent's decision tree.
        
        Parameters
        --------------
        
        Returns
        --------------
        nltk tree object, helpful to visualize the agent's decision tree policy.
        """

        return Tree.fromstring(self.build_string(encoder))
        
        
            
            
        
        
            
            
