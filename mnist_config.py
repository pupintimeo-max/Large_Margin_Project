import numpy as np
import tensorflow as tf

class ConfigDict:
    """MNIST configuration."""

    def __init__(self):
        # Optimization parameters.
        self.init_lr = 0.01
        self._l2_loss_wt = 5e-4
        self.xent_loss_wt = 1.0
        self.margin_loss_wt = 0.0
        self.gamma = 10000
        self.alpha = 4
        self.top_k = 1
        self.dist_norm = np.inf
        self.num_classes = 10

        # List of tuples specify (kernel_size, number of filters) for each layer.
        self.filter_sizes_conv_layers = [(5, 32), (5, 64)]
        # Dictionary of pooling type ("max"/"average", size and stride).
        self.pool_params = {"type": "max", "size": 2, "stride": 2}
        self.num_units_fc_layers = [512]
        self.dropout_rate = 0
        self.batch_norm = True
        self.activation = tf.nn.relu
        self._regularizer = tf.keras.regularizers.L2(self._l2_loss_wt)

    @property
    def l2_loss_wt(self):
        return self._l2_loss_wt

    @l2_loss_wt.setter
    def l2_loss_wt(self, value):
        self._l2_loss_wt = value
        self._regularizer = tf.keras.regularizers.L2(self._l2_loss_wt)

    @property
    def regularizer(self):
        return self._regularizer