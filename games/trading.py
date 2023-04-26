import numpy as np
from .abstract_game import AbstractGame
import requests
import random
import datetime
from datetime import timedelta
import pathlib
import torch
import os
import json

max_bars = 500
starting_balance = 100000
stake_amount = 100
class MuZeroConfig:
    def __init__(self):
        self.seed = 0
        self.max_moves = int(max_bars / 2)
        self.num_simulations = 100
        self.discount = 0.997
        self.temperature_threshold = 0
        self.exploration_fraction = 0.25
        self.checkpoint_interval = int(1e3)
        self.num_actors = 4
        self.lr_init = 0.1
        self.lr_decay_rate = 0.1
        self.window_size = int(1e6)
        self.batch_size = 512
        self.num_unroll_steps = 5
        self.td_steps = self.max_moves        
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available
        self.selfplay_on_gpu = False
        
        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25
        
        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        
        ### Game
        self.observation_shape = (1, 1, self.max_moves)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(3))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = self.max_moves  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 10000 # Total number of training steps (ie weights update according to a batch)
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 1  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available
        
        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        self.reanalyse_on_gpu = False
        
        ### Network
        self.network = "fullyconnected"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        self.use_batch_norm = True
        
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


        # Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.num_actors = 3000
        self.max_moves = self.max_moves * self.num_actors
        self.num_simulations = 50
        self.self_play_delay = 2
        self.temperature_threshold = 15

        ### Replay Buffer
        self.replay_buffer_size = 3000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 20  # Number of game moves to keep for every batch element
        self.td_steps = 20  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on
        
        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1
    
class TradingEnv:
    def __init__(self, seed=None):
        """
        Initialize Trading Battle game.

        Args:
            seed: Random seed.
        """
        self.seed = seed
        self.pair = "BTCUSDT"
        self.balance = starting_balance
        self.position = 0
        self.avg_price = 0
        self.observation = None
        self.current_step = 0
        self.max_steps = int(max_bars / 2)
        self.prices = None
        self.prices_length = int(max_bars / 2)

    def generate_prices(self, start_date, end_date):
        # Define cache directory
        cache_dir = "binance_cache"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Define cache filename based on start and end dates
        cache_filename = f"{self.pair}_{start_date}_{end_date}.json"
        cache_filepath = os.path.join(cache_dir, cache_filename)

        if os.path.exists(cache_filepath):
            # Load cached data from file
            with open(cache_filepath, "r") as cache_file:
                close_prices = json.load(cache_file)
            print(f"Retrieved {len(close_prices)} candles from cache for {start_date} to {end_date}\n")
        else:
            # Define Binance API endpoint for klines (candlestick data)
            api_url = "https://api.binance.com/api/v3/klines"

            # Define interval as 1 minute
            interval = "1m"

            # Define date format for API request
            date_format = "%Y-%m-%d %H:%M:%S"

            # Define start and end timestamps in milliseconds
            start_timestamp = int(datetime.datetime.strptime(str(start_date), date_format).timestamp() * 1000)
            end_timestamp = int(datetime.datetime.strptime(str(end_date), date_format).timestamp() * 1000)

            # Define query parameters for API request
            query_params = {
                "symbol": self.pair,
                "interval": interval,
                "startTime": start_timestamp,
                "endTime": end_timestamp
            }

            # Send API request and retrieve response as a JSON object
            response = requests.get(api_url, params=query_params).json()

            # Extract candlestick data from response
            close_prices = []
            for candlestick in response:
                close_price = float(candlestick[4])
                close_prices.append(close_price)

            # Return candlestick data
            print(f"Binance Download at {datetime.datetime.now()} Retrieved {len(close_prices)} candles from {start_date} to {end_date}\n")
        return close_prices

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        
        print(f"STEP {self.current_step} balance:{self.balance} position:{self.position}")
        
        assert action in self.legal_actions(), "Invalid action"
        
        price = self.prices[self.current_step]
        if action == 1:  # Buy
            if (self.position < 0):
                #close a short position
                quantity = self.position
                self.balance += (abs(self.position) * self.avg_price) - (abs(self.position) * price) 
                self.avg_price = 0
                self.position = 0
            else: 
                quantity = (stake_amount / price)
                self.avg_price = ((self.position * self.avg_price) + (quantity * price)) / (quantity + self.position)
                self.position += quantity
                self.balance -= quantity * price
            self.balance -= (quantity * price * 0.001)
        else:  # Sell
            if (self.position > 0):
                #close a long position
                quantity = self.position
                self.balance += (self.position * price) - (self.position * self.avg_price)
                self.avg_price = 0
                self.position = 0
            else: 
                quantity = (stake_amount / price)
                self.avg_price = ((-self.position * self.avg_price) + (quantity * price)) / (quantity + -self.position)
                self.position -= quantity
                self.balance -= quantity * price
            self.balance -= (quantity * price * 0.001)

        reward =  (self.position * (self.avg_price - price)) + (self.balance - starting_balance)
        
        self.current_step += 1
        self.observation = self.prices[self.current_step:self.current_step+self.prices_length]
        done = self.current_step >= self.max_steps or reward <= -starting_balance

        return self.observation, reward, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return 0

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        if self.balance < stake_amount:            
            return np.array([0]) if self.position == 0 else np.array([0, (2 if self.position > 0 else 1)])

        return np.array([0, 1, 2])

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        self.balance = starting_balance
        self.current_step = 0
        self.position = 0
        self.avg_price = 0
        
        rng = random.Random(self.seed)
        start_date = datetime.datetime(year=rng.randint(2021, 2022), month=rng.randint(1, 12), day=rng.randint(1, 28))
        end_date = start_date + timedelta(minutes=self.max_steps + self.prices_length + 1)

        self.prices = self.generate_prices(start_date, end_date)
        self.observation = self.prices[self.current_step:self.current_step+self.prices_length]

        return [[self.observation]]

    def close(self):
        """
        Properly close the game.
        """
        pass

    def render(self):
        """
        Display the game observation.
        """
        print("Balance:", self.balance)
        print("Prices:", self.prices[self.current_step:self.current_step+self.prices_length])

    def set_prices(self, prices):
        """
        Set the prices for the game.

        Args:
            prices: A 1D numpy array with prices.
        """
        self.prices = prices
        self.prices_length = len(prices) - self.max_steps


class Game(AbstractGame):
    def __init__(self, seed=None):
        self.env = TradingEnv(seed)

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
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

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
            1: "Long",
            2: "Short",
        }
        return f"{action_number}. {actions[action_number]}"
