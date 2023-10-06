import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the Linear_QNet neural network.

        Parameters:
        - input_size (int): The number of input neurons.
        - hidden_size (int): The number of neurons in the hidden layer.
        - output_size (int): The number of output neurons representing Q-values for actions.
        """
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Parameters:
        - x (torch.Tensor): The input state tensor.

        Returns:
        - torch.Tensor: The output tensor representing Q-values for actions.
        """
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model.pth'):
        """
        Saves the model's state dictionary to a file.

        Parameters:
        - file_name (str): The name of the file to save the model weights (default: 'model.pth').
        """
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        """
        Initializes the QTrainer.

        Parameters:
         model (Linear_QNet): The Q-network model to be trained.
        - lr (float): The learning rate for the optimizer.
        - gamma (float): The discount factor for future rewards.
        """
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)

        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """
        Performs a single training step of the Q-network.

        Parameters:
        - state (numpy.ndarray or list): The current state of the game.
        - action (int): The action taken in the current state.
        - reward (float): The immediate reward received after taking the action.
        - next_state (numpy.ndarray or list): The next state resulting from the action.
        - done (bool): Indicates whether the game episode is complete (True) or not (False).
        """
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1. Predicted Q values with current state
        pred =  self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            target[idx][torch.argmax(action).item()] = Q_new
        

        # 2. Q_new = r+y*max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()