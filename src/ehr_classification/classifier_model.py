from tensorflow.keras.layers import Conv1D, LSTM
from tensorflow.keras.layers import Input, Dropout, MaxPooling1D, concatenate, Embedding, Bidirectional, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop


# multi_convolution module for the neural network
def multi_conv(x, num_kernel, activation='relu'):
    kreg = None
    a = Conv1D(num_kernel, 3, activation=activation, padding='valid', kernel_regularizer=kreg)(x)
    b = Conv1D(num_kernel, 3, activation=activation, padding='same', kernel_regularizer=kreg)(x)
    b = Conv1D(num_kernel, 3, activation=activation, padding='valid', kernel_regularizer=kreg)(b)
    return concatenate([a, b], axis=-1)


# the structure of model
def compile_lstm(embeddings, shape, settings):
    input1 = Input((shape['max_length'],))
    initial_kernel_num = 64

    x = input1
    x = Embedding(embeddings.shape[0], embeddings.shape[1], input_length=shape['max_length'], trainable=False,
                  weights=[embeddings], mask_zero=False, )(x)
    x = Bidirectional(LSTM(initial_kernel_num * 2, return_sequences=True))(x)
    x = Conv1D(initial_kernel_num, 5, activation='relu', padding='valid')(x)
    x = multi_conv(x, initial_kernel_num)
    x = Bidirectional(LSTM(initial_kernel_num * 2, return_sequences=True))(x)
    x = MaxPooling1D(pool_size=3)(x)
    x = multi_conv(x, initial_kernel_num * 2)
    x = Bidirectional(LSTM(initial_kernel_num * 4, return_sequences=True))(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = multi_conv(x, initial_kernel_num * 4)
    x = Bidirectional(LSTM(initial_kernel_num * 8, return_sequences=True))(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = multi_conv(x, initial_kernel_num * 4)
    x = Bidirectional(LSTM(initial_kernel_num * 8, return_sequences=True))(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = multi_conv(x, initial_kernel_num * 4)
    x = Bidirectional(LSTM(initial_kernel_num * 8, return_sequences=True))(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = multi_conv(x, initial_kernel_num * 4)
    x = Bidirectional(LSTM(initial_kernel_num * 8, return_sequences=True))(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = multi_conv(x, initial_kernel_num * 4)
    x = Bidirectional(LSTM(initial_kernel_num * 8, return_sequences=True))(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = multi_conv(x, initial_kernel_num * 4)
    x = Bidirectional(LSTM(initial_kernel_num * 8, return_sequences=False))(x)
    a1 = Dense(512, activation='relu')(x)
    a1 = Dropout(0.5)(a1)
    a1 = Dense(1, name='CAT')(a1)
    b1 = Activation('sigmoid')(a1)
    model = Model(inputs=input1, outputs=[b1, a1])
    loss = ['binary_crossentropy', 'mean_squared_error']
    model.compile(optimizer=RMSprop(lr=settings['lr']), loss=loss, loss_weights=[1.0, 0.0], metrics=['accuracy'])
    return model
