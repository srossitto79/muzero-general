import numpy as np
from typing import List, Tuple
from .abstract_game import AbstractGame

class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        # pairs: BTC/USDT, ETH/USDT, ADA/USDT, BNB/USDT, XRP/USDT, LTC/USDT
        self.observation_shape = (1, 5, 60)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(3))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = 120  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 1440  # Maximum number of moves if game is not finished before
        self.num_simulations = 10  # Number of future moves self-simulated
        self.discount = 0.978  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "fullyconnected"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        
        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 2  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 2  # Number of channels in policy head
        self.resnet_fc_reward_layers = []  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = []  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = []  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 5
        self.fc_representation_layers = [16]  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = [16]  # Define the hidden layers in the value network
        self.fc_policy_layers = [16]  # Define the hidden layers in the policy network



        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 30000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 32  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 1  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.0064  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 1000



        ### Replay Buffer
        self.replay_buffer_size = 5000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 7  # Number of game moves to keep for every batch element
        self.td_steps = 7  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0.2  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1



# This implementation simulates a game in which the player can buy or sell a single unit of each of a set of assets 
# at each time step, with transaction costs applied. The observation is the price history of each asset up to the 
# current time step, and the reward is the difference between the current portfolio value and the initial portfolio value.
# The legal actions are simply buying or selling each of the assets. You would need to pass in historical price data as a 
# numpy array to the price_history argument when creating an instance of the game.

class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = TradingEnv()

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return [[observation]], reward * 10, done

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return list(range(3))

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return [[self.env.reset()]]

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        #input("Press enter to take a step ")

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        actions = {
            0: "Hold",
            1: "Buy",
            1: "Sell",
        }
        return f"{action_number}. {actions[action_number]}"

class TradingEnv:
    def __init__(self, pair = 'BTC/USDT', balance = 1000):
        self.position = 0 #hold
        self.pair = pair
        self.balance = balance
        self.start_balance = balance
        self.orders_amount = balance * 0.05
        self.avg_price = 0

    def legal_actions(self):
        legal_actions = list(range(3))
        return legal_actions

    def step(self, action):
        #TODO
        #when action is 1-buy we increase the position by orders_amount and update the average entry price if needed
        #when action is 2-sell we decrease the position by orders_amount and update the average entry price if needed
        #we calc reward as the (current_price - avg_price) * self.position 
        
        #TODO_UPDATE_PRICE
        
        if action not in self.legal_actions():
            pass
        elif action == 1:
            self.position += self.orders_amount
        elif action == 2:
            self.position -= self.orders_amount

        reward = self.avg_price
        return self.get_observation(), reward, bool(reward)

    def reset(self):
        self.position = [0, 0]
        return self.get_observation()

    def render(self):
        im = numpy.full((self.size, self.size), "-")
        im[self.size - 1, self.size - 1] = "1"
        im[self.position[0], self.position[1]] = "x"
        print(im)

    def get_observation(self):
        # todo, lets put a bool training_mode, 
        # when true we grab random dataframes from binance, 60 1-minute bars on a random epoch on the same pair. 
        # when false, we get the last available 60 bars from rt data
        observation = numpy.zeros((self.size, self.size))
        observation[self.position[0]][self.position[1]] = 1
        return observation.flatten()

class Game(AbstractGame):
    def __init__(self, initial_portfolio: List[float], price_history: np.ndarray, transaction_cost: float = 0.001):
        """
        Args:
            initial_portfolio (List[float]): initial portfolio of assets
            price_history (np.ndarray): historical price data for each asset
            transaction_cost (float, optional): cost of transactions as a percentage of the transaction value. Defaults to 0.001.
        """
        
        self.initial_portfolio = np.array(initial_portfolio)
        self.price_history = price_history
        self.transaction_cost = transaction_cost
        self.current_step = 0
        self.current_portfolio = self.initial_portfolio.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Perform a step in the game.

        Args:
            action (int): index of the asset to buy or sell. If positive, buy the asset. If negative, sell the asset.

        Returns:
            Tuple[np.ndarray, float, bool, dict]: tuple containing the new observation, reward, done flag, and info dictionary.
        """
        transaction_value = self.current_portfolio[action] * self.price_history[self.current_step, action]
        transaction_cost = transaction_value * self.transaction_cost
        transaction_net = transaction_value - transaction_cost
        self.current_portfolio[action] += 1 if action >= 0 else -1
        self.current_portfolio -= transaction_net / self.price_history[self.current_step]
        self.current_step += 1

        observation = self.get_observation()
        reward = self.get_reward()
        done = self.current_step == self.price_history.shape[0]
        info = {}

        return observation, reward, done, info

    def get_observation(self) -> np.ndarray:
        """
        Get the current observation.

        Returns:
            np.ndarray: the current observation.
        """
        return self.price_history[self.current_step]

    def get_reward(self) -> float:
        """
        Get the current reward.

        Returns:
            float: the current reward.
        """
        return np.sum(self.current_portfolio * self.price_history[self.current_step]) - np.sum(self.initial_portfolio * self.price_history[0])

    def get_legal_actions(self) -> List[int]:
        """
        Get the legal actions for the current state.

        Returns:
            List[int]: a list of legal actions.
        """
        return list(range(len(self.initial_portfolio)))
