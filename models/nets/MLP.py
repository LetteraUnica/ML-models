from torch import nn

class BatchnormedLayer(nn.Module):
    """Single MLP layer with batchnorm and dropout
    dropout is applied before the activation function
    and batchnorm is applied after the activation function"""
    def __init__(self, in_features, out_features, dropout=0.1,
                 batchnorm=True, activation=nn.ReLU()):
        super().__init__()
        
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_features, out_features))
        layers.append(activation)
        if batchnorm:
            layers.append(nn.BatchNorm1d(out_features))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class MLPBlock(nn.Module):
    """Sequence of n linear layers with dropout, batchnorm
    and a skip connection from input to output.
    Analogue of the ResNet convolutional block"""
    def __init__(self, hidden_size,
                 hidden_layers=2, dropout=0.1,
                 skip=True, batchnorm=True, activation=nn.ReLU()):
        super().__init__()

        self.skip = skip

        self.layers = nn.Sequential(
            *[BatchnormedLayer(hidden_size, hidden_size, dropout, batchnorm, activation) for _ in range(hidden_layers)]
        )

    def forward(self, x):
        if self.skip:
            return x + self.layers(x)
        return self.layers(x)


class MLP(nn.Module):
    """MLP composed of n MLPBlocks, between each block there is a skip connection"""
    def __init__(self, in_features, out_features, hidden_size=512,
                 mlp_blocks=3, hidden_layers=1, dropout=0.1,
                 skip=True, batchnorm=True, activation=nn.ReLU()):
        super().__init__()

        self.input_layer = nn.Sequential(*[nn.Linear(in_features, hidden_size)])

        self.layers = nn.Sequential(
            *[MLPBlock(hidden_size, hidden_layers, dropout, skip, batchnorm, activation) for _ in range(mlp_blocks)]
        )

        self.output_layer = nn.Sequential(*[nn.Linear(hidden_size, out_features)])

    def forward(self, x):
        x = self.input_layer(x)
        x = self.layers(x)
        x = self.output_layer(x)
        return x