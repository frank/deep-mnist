import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
Omits the Tensorflow warning messages at runtime.
The warnings concern the CPU version of tf which I'm not using.
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

num_nodes_hl1 = 1000
num_nodes_hl2 = 1000
num_nodes_hl3 = 1000

num_classes = 10
batch_size = 100

'''
Placeholders that will contain the input samples and
their labels, respectively.
'''
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


def neural_network_model(data):
    """Neural network model.
    Is made of two hidden layers and an output layer, where
    each layer contains the weights matrix and the biases array.
    All layers use the rectifier function as their activation function.
    """
    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([784, num_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([num_nodes_hl1]))}

    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([num_nodes_hl1, num_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([num_nodes_hl2]))}

    hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([num_nodes_hl2, num_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([num_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([num_nodes_hl3, num_classes])),
                    'biases': tf.Variable(tf.random_normal([num_classes]))}

    hl1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    hl1 = tf.nn.relu(hl1)

    hl2 = tf.add(tf.matmul(hl1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    hl2 = tf.nn.relu(hl2)

    hl3 = tf.add(tf.matmul(hl2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    hl3 = tf.nn.relu(hl3)

    output = tf.add(tf.matmul(hl3, output_layer['weights']), output_layer['biases'])

    return output


def train_neural_network(x):
    """Trainer for the neural network.
    The network's execution, its loss (how good it performs compared
    to the expected outcome) and the optimizer function are declared.
    Then the training of the network takes place within the tensorflow session."""

    prediction = neural_network_model(x)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    hm_epochs = 10

    with tf.Session() as sess:
        # Initializes all variables of the model.
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):

            # Represents the amount of errors that the network made in the current epoch.
            epoch_loss = 0

            # Train the network in batches of 100 samples at a time, calculating the loss at each epoch.
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, loss], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, '. Loss:', epoch_loss)

        # Printing the accuracy of the network at the end of the training process.
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train_neural_network(x)
