import numpy as np
import tensorflow as tf 

class NeuralTuringMachine(tf.keras.Model):
    def __init__(self, num_inputs, num_outputs, memory_size, memory_vector_dim):
        super(NeuralTuringMachine, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs 
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim 
        
        self.memory = tf.Variable(tf.zeros([memory_size, memory_vector_dim]))

    def call(self, inputs):
        # Example call structure - customize further based on expected outputs and read/write mechanisms
        output = tf.zeros([tf.shape(inputs)[0], self.num_outputs])  # Placeholder for actual processing logic
        return output


class NTMController(tf.keras.layers.Layer):
    def __init__(self, num_inputs, num_outputs, memory_vector_dim):
        super(NTMController, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(num_outputs + memory_vector_dim * 2)
        
    def call(self, inputs, prev_state):
        x = tf.concat([inputs, prev_state], axis=-1)
        x = self.dense1(x)
        return self.dense2(x)
    
def cosine_similarity(x, y):
    return tf.reduce_sum(x * y, axis=-1) / (tf.norm(x, axis=-1) * tf.norm(y, axis=-1) + 1e-8)

def content_addressing(key, memory):
    similarity = cosine_similarity(key[:, tf.newaxis, :], memory)
    return tf.nn.softmax(similarity, axis=-1)
    
def read_memory(memory, read_weights):
    return tf.reduce_sum(memory * read_weights[:, :, tf.newaxis], axis=1)

def write_memory(memory, write_weights, erase_vector, add_vector):
    erase = tf.reduce_sum(write_weights[:, :, tf.newaxis] * erase_vector[:, tf.newaxis, :], axis=1)
    add = tf.reduce_sum(write_weights[:, :, tf.newaxis] * add_vector[:, tf.newaxis, :], axis=1)
    return memory * (1 - erase) + add

@tf.function
def train_step(inputs, targets, model, optimizer):
    with tf.GradientTape() as tape:
        outputs = model(inputs)
        loss = tf.reduce_mean(tf.square(outputs - tf.cast(targets, tf.float32)))
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss 

def generate_task(sequence_length, vector_dim):
    sequence = np.random.randint(0, 2, size=(1, sequence_length, vector_dim))
    inputs = np.concatenate([sequence, np.zeros((1, 1, vector_dim))], axis=1)
    targets = np.concatenate([np.zeros_like(sequence), sequence], axis=1)
    return inputs, targets


def attention_mechanism(query, keys, values):
    d_k = tf.sqrt(tf.cast(tf.shape(keys)[-1], tf.float32) + 1e-8)  # Prevent divide by zero
    attention_weights = tf.nn.softmax(tf.matmul(query, keys, transpose_b=True) / d_k)
    return tf.matmul(attention_weights, values)

def read_with_attention(memory, read_query):
    read_attention = attention_mechanism(read_query, memory, memory)
    return tf.reduce_sum(read_attention, axis=1)

def compute_gradients(model, inputs, targets):
    with tf.GradientTape() as tape:
        outputs = model(inputs)
        loss = tf.reduce_mean(tf.square(outputs - targets))
    return tape.gradient(loss, model.trainable_variables)

lstm = tf.keras.layers.LSTM(64, return_sequences=True)

class SimpleNTM(tf.keras.layers.Layer):
    def __init__(self, units, memory_size, memory_vector_dim):
        super(SimpleNTM, self).__init__()
        self.controller = tf.keras.layers.LSTMCell(units)
        self.memory = tf.Variable(tf.zeros([memory_size, memory_vector_dim]))
        
    def call(self, inputs, states):
        output, new_states = self.controller(inputs, states)
        read_vector = read_with_attention(self.memory, output)
        return tf.concat([output, read_vector], axis=-1), new_states
    

def generate_sorting_task(sequence_length, max_value):
    sequence = np.random.randint(0, max_value, size=(1, sequence_length))
    inputs = np.eye(max_value)[sequence]
    targets = np.eye(max_value)[np.sort(sequence)]
    return inputs, targets

optimizer = tf.keras.optimizers.Adam()  # Initialize optimizer here

def train_sorting_ntm(ntm, num_epochs):
    for epoch in range(num_epochs):
        inputs, targets = generate_sorting_task(10, 20)
        loss = train_step(inputs, targets, ntm, optimizer)
        if epoch % 100 == 0:
            print(f"epoch {epoch}, loss: {loss.numpy()}")

ntm_sorter = NeuralTuringMachine(20, 20, 128, 32)
train_sorting_ntm(ntm_sorter, 100)
