from tensorflow.keras import layers, Model

class TextLargeMarginModel(Model):
    def __init__(self, vocab_size=10000, embed_dim=50, num_classes=2, **kwargs):
        super(TextLargeMarginModel, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.embedding_layer = layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim)

        self.pool = layers.GlobalAveragePooling1D()

        self.fc = layers.Dense(64, activation="relu")

        self.classifier = layers.Dense(self.num_classes, activation=None)

    def call(self, inputs):
        # inputs est un tenseur d'entiers [Batch, Sequence_Length]

        # On récupère les vecteurs (C'est notre "Image" pour le texte)
        embedded_x = self.embedding_layer(inputs)

        pool_x = self.pool(embedded_x)
        fc_x = self.fc(pool_x)
        logits = self.classifier(fc_x)

        # IMPORTANT : On retourne (logits, embedded_x)
        # embedded_x servira à calculer la marge
        return logits, [embedded_x, pool_x, fc_x]

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_classes": self.num_classes,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)