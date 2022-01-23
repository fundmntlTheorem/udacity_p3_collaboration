from collections import OrderedDict
from utilities.network import*

class FCBody(nn.Module):
    '''
        This is a fully connected body network, which contains an input
        layer and several hidden layers.
        input_dim : scalar int, number of nodes in input layer
        body_dim : tuple of int, number of nodes in each hidden layer
        gate_func : activation function applied after each layer
    '''
    def __init__(self, input_dim, body_dim=(400, 300), gate_func=nn.ReLU):
        super().__init__()
        dims = (input_dim,) + body_dim
        layers = []
        for idx,(input,output) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append((f'layer{idx}', layer_init(nn.Linear(input, output))))
            layers.append((f'gate{idx}', gate_func()))     

        self.model = nn.Sequential(OrderedDict(layers))
        self.gate = gate_func
        self.feature_dim = dims[-1]

    def forward(self, x):
        return self.model(x)

class Actor(nn.Module):
    def __init__(self, body, output_dim, output_gate_func=empty_gate, weight_bounds=(-3e-3, 3e-3)):
        super().__init__()
        self.body = body
        self.head = layer_init(nn.Linear(body.feature_dim, output_dim), weight_bounds)
        self.output_gate_func = output_gate_func()

    def forward(self, x):
        return self.output_gate_func(self.head(self.body(x)))

class Critic(nn.Module):
    """Critic (Value) Model.
        From Udacity\deep-reinforcement-learning\ddpg-pendulum
    """

    def __init__(self, state_size, action_size, fcs1_units=400, fc2_units=300, weight_bounds=(-3e-3, 3e-3)):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super().__init__()
        self.fcs1 = layer_init(nn.Linear(state_size, fcs1_units))
        self.fc2 = layer_init(nn.Linear(fcs1_units+action_size, fc2_units))
        self.fc3 = layer_init(nn.Linear(fc2_units, 1), weight_bounds)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)