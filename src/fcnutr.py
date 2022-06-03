from tensorflow import keras

class FCNutr(keras.Model):
    def __init__(self, nutr_names, crop_size, hidden_dim, num_shared_hidden, use_dropout):
        super().__init__()
        
        self.cnn = keras.applications.InceptionV3(
            include_top=False, 
            weights="imagenet", 
            input_shape=crop_size + (3,), 
            pooling='avg'
        )

        shared_layers = []
        for i in range(num_shared_hidden):
            if use_dropout:
                shared_layers.append(keras.layers.Dropout(0.2 if i == 0 else 0.5))
            shared_layers.append(keras.layers.Dense(hidden_dim, activation="relu"))
            
        self.shared = keras.Sequential(shared_layers, name='shared')

        self.multitask_heads = []
        for name in nutr_names:
            head_layers = [
                keras.layers.Dense(hidden_dim, activation="relu"),
                keras.layers.Dense(1, activation="relu")
            ]

            if use_dropout:
                head_layers.insert(0, keras.layers.Dropout(0.5))
                head_layers.insert(2, keras.layers.Dropout(0.5))

            self.multitask_heads.append(keras.Sequential(head_layers, name=name))

    def call(self, inputs, training=False):
        x = self.cnn(inputs, training=training)
        x = self.shared(x, training=training)
        return {head.name: head(x, training=training) for head in self.multitask_heads}
