import tensorflow as tf
from mnist_config import ConfigDict

MOMENTUM = 0.9
EPS = 1e-5

class MNISTNetwork(tf.keras.Model):
    """MNIST model using Keras subclassing API."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # Store the config object, as it contains all parameters
        self._config = config

        self.num_classes = config.num_classes
        self.activation = config.activation
        self.filter_sizes_conv_layers = config.filter_sizes_conv_layers
        self.num_units_fc_layers = config.num_units_fc_layers
        self.pool_params = config.pool_params
        self.dropout_rate = config.dropout_rate
        self.use_batch_norm = config.batch_norm
        self.regularizer = config.regularizer

        # Build convolutional layers
        self.conv_layers = []
        self.pool_layers = []
        self.bn_layers = []
        self.dropout_layers = []

        for i, (kernel_size, filters) in enumerate(self.filter_sizes_conv_layers):
            self.conv_layers.append(
                tf.keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=(1, 1),
                    padding="same",
                    activation=None,
                    kernel_regularizer=self.regularizer,
                    use_bias=not self.use_batch_norm,
                    name=f"conv_layer{i}",
                )
            )

            if self.pool_params:
                if self.pool_params["type"] == "max":
                    self.pool_layers.append(
                        tf.keras.layers.MaxPooling2D(
                            pool_size=self.pool_params["size"],
                            strides=self.pool_params["stride"],
                            name=f"pool_layer{i}",
                        )
                    )
                else:
                    self.pool_layers.append(
                        tf.keras.layers.AveragePooling2D(
                            pool_size=self.pool_params["size"],
                            strides=self.pool_params["stride"],
                            name=f"pool_layer{i}",
                        )
                    )
            else:
                self.pool_layers.append(None)

            if self.dropout_rate > 0:
                self.dropout_layers.append(
                    tf.keras.layers.Dropout(
                        rate=self.dropout_rate, name=f"dropout_layer{i}"
                    )
                )
            else:
                self.dropout_layers.append(None)

            if self.use_batch_norm:
                self.bn_layers.append(
                    tf.keras.layers.BatchNormalization(
                        momentum=MOMENTUM, epsilon=EPS, name=f"bn_layer{i}"
                    )
                )
            else:
                self.bn_layers.append(None)

        # Flatten layer
        self.flatten = tf.keras.layers.Flatten()

        # Fully connected layers
        self.fc_layers = []
        for i, num_units in enumerate(self.num_units_fc_layers):
            self.fc_layers.append(
                tf.keras.layers.Dense(
                    num_units,
                    activation=self.activation,
                    kernel_regularizer=self.regularizer,
                    use_bias=True,
                    name=f"fc_layer{i}",
                )
            )

        # Output layer
        self.output_layer = tf.keras.layers.Dense(
            self.num_classes,
            activation=None,
            kernel_regularizer=self.regularizer,
            name="output_layer",
        )

    def call(self, inputs, training=False):
        """Forward pass returning logits and intermediate layer outputs.

        Args:
            inputs: Input tensor of shape [batch, height, width, channels].
            training: Boolean indicating training mode.

        Returns:
            logits: Output logits tensor.
            endpoints: Dictionary of intermediate layer outputs.
        """
        endpoints = {}
        net = inputs

        for i, (conv, pool, dropout, bn) in enumerate(
            zip(self.conv_layers, self.pool_layers, self.dropout_layers, self.bn_layers)
        ):
            net = conv(net)
            net = self.activation(net)

            if pool is not None:
                net = pool(net)

            if dropout is not None:
                net = dropout(net, training=training)

            if bn is not None:
                net = bn(net, training=training)

            endpoints[f"conv_layer{i}"] = net

        net = self.flatten(net)

        for i, fc in enumerate(self.fc_layers):
            net = fc(net)
            endpoints[f"fc_layer{i}"] = net

        logits = self.output_layer(net)
        endpoints["logits"] = net  # Pre-logits features

        return logits, endpoints

    def get_config(self):
        config_dict = super().get_config()
        # Explicitly serialize all relevant attributes of the internal _config object
        config_dict.update({
            "init_lr": self._config.init_lr,
            "l2_loss_wt": self._config.l2_loss_wt,
            "xent_loss_wt": self._config.xent_loss_wt,
            "margin_loss_wt": self._config.margin_loss_wt,
            "gamma": self._config.gamma,
            "alpha": self._config.alpha,
            "top_k": self._config.top_k,
            "dist_norm": self._config.dist_norm,
            "num_classes": self._config.num_classes,
            "filter_sizes_conv_layers": self._config.filter_sizes_conv_layers,
            "pool_params": self._config.pool_params,
            "num_units_fc_layers": self._config.num_units_fc_layers,
            "dropout_rate": self._config.dropout_rate,
            "batch_norm": self._config.batch_norm, # Use 'batch_norm' directly as it's the ConfigDict attribute
        })
        return config_dict

    @classmethod
    def from_config(cls, config_dict):
        # Reconstruct the ConfigDict object from the flattened config_dict
        reconstructed_config = ConfigDict()
        reconstructed_config.init_lr = config_dict.get("init_lr", reconstructed_config.init_lr)
        reconstructed_config.l2_loss_wt = config_dict.get("l2_loss_wt", reconstructed_config.l2_loss_wt)
        reconstructed_config.xent_loss_wt = config_dict.get("xent_loss_wt", reconstructed_config.xent_loss_wt)
        reconstructed_config.margin_loss_wt = config_dict.get("margin_loss_wt", reconstructed_config.margin_loss_wt)
        reconstructed_config.gamma = config_dict.get("gamma", reconstructed_config.gamma)
        reconstructed_config.alpha = config_dict.get("alpha", reconstructed_config.alpha)
        reconstructed_config.top_k = config_dict.get("top_k", reconstructed_config.top_k)
        reconstructed_config.dist_norm = config_dict.get("dist_norm", reconstructed_config.dist_norm)
        reconstructed_config.num_classes = config_dict.get("num_classes", reconstructed_config.num_classes)
        reconstructed_config.filter_sizes_conv_layers = config_dict.get("filter_sizes_conv_layers", reconstructed_config.filter_sizes_conv_layers)
        reconstructed_config.pool_params = config_dict.get("pool_params", reconstructed_config.pool_params)
        reconstructed_config.num_units_fc_layers = config_dict.get("num_units_fc_layers", reconstructed_config.num_units_fc_layers)
        reconstructed_config.dropout_rate = config_dict.get("dropout_rate", reconstructed_config.dropout_rate)
        reconstructed_config.batch_norm = config_dict.get("batch_norm", reconstructed_config.batch_norm)

        # Instantiate the model by passing the reconstructed ConfigDict object
        return cls(reconstructed_config)