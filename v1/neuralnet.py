from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def build_model(input_shape, learning_rate):
    model = Sequential()

    model.add(InputLayer(input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=16, kernel_size=4, activation="relu", kernel_regularizer=l2(1e-3)))
    model.add(Conv2D(filters=16, kernel_size=4, activation="relu", kernel_regularizer=l2(1e-3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))

    model.add(Conv2D(filters=32, kernel_size=4, activation="relu", kernel_regularizer=l2(1e-3)))
    model.add(Conv2D(filters=32, kernel_size=4, activation="relu", kernel_regularizer=l2(1e-3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))
    
    model.add(Conv2D(filters=32, kernel_size=4, activation="relu", kernel_regularizer=l2(1e-3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=256, activation="relu"))
    model.add(BatchNormalization())

    model.add(Dense(units=128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(28, activation="softmax"))

    """
    optimizer = Adam(learning_rate=learning_rate, decay=1e-5)
    model.compile(optimizer=optimizer, 
        loss="categorical_crossentropy", metrics=["accuracy"]
    )
    """

    return model


if __name__ == "__main__":
    m = build_model((128, 251, 1), learning_rate=1e-3)
    m.summary()
