import numpy as np
import pandas as pd
import tensorflow as tf
import h5py
import time

from Data_Reader import windowz, segment_opp, data_reader

# reset tensorflow graph
tf.reset_default_graph()

# Class defining values for LSTM
class Config(object):
    def __init__(self, X_train, X_test, input_width):
        # Input data
        self.train_count = len(X_train)  # 7352 training series
        self.test_data_count = len(X_test)  # 2947 testing series
        self.n_steps = len(X_train[0])  # 128 time_steps per series
        print("len(x_train[0])",len(X_train[0]))
        
        print("For Opportunity Data!")
        #self.input_height = 1
        self.input_width = input_width #or 90 for actitracker
        self.num_labels = 18  #or 6 for actitracker
        self.num_channels = 77 #or 3 for actitracker 

        self.learning_rate = 0.001
        self.lambda_loss_amount = 0.0015
        self.training_epochs = 10
        self.batch_size = 64

        # LSTM structure
        self.n_inputs = len(X_train[0][0])  # Features count is of 9: 3 * 3D sensors features over time
        print("n_inputs len(X_train[0][0])",len(X_train[0][0]))
        self.n_hidden = 64  # nb of neurons inside the neural network
        self.n_classes = self.num_labels  # Final output classes
        self.W = {
            'hidden': tf.Variable(tf.random_normal([self.n_inputs, 2*self.n_hidden])),
            'output': tf.Variable(tf.random_normal([2*self.n_hidden, self.n_classes]))
        }
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([2*self.n_hidden], mean=1.0)),
            'output': tf.Variable(tf.random_normal([self.n_classes]))
        }
        

# LSTM Structure
def LSTM_Network(_X, config):
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    _X = tf.reshape(_X, [-1, config.n_inputs])

    # Linear activation
    _X = tf.nn.relu(tf.matmul(_X, config.W['hidden']) + config.biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, config.n_steps, 0)
    
    #_X = tf.unstack(_X, config.n_steps, 1)
    
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=0.5)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=0.5)
    
    try:
        outputs, _,_ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, _X, dtype=tf.float32)
    except Exception:
        outputs = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, _X, dtype=tf.float32)
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, config.W['output']) + config.biases['output']


if __name__ == "__main__":
    
    # reading the saved h5 data
    h5_path = ''
    #dr = data_reader(h5_path)
    f = h5py.File('opportunity.h5', 'r')

    x_train = f.get('train').get('inputs')[()]
    y_train = f.get('train').get('targets')[()]

    x_test = f.get('test').get('inputs')[()]
    y_test = f.get('test').get('targets')[()]
    
    # close h5 file
    f.close()
    
    # create segments
    input_width = 24
    print("segmenting signal...")
    train_x, train_y = segment_opp(x_train,y_train,input_width)
    test_x, test_y = segment_opp(x_test,y_test,input_width)
    print("signal segmented!")

    print("")
    print("Printing shapes of training and testing data:")
    print("train shape:", train_x.shape)
    print("train - label shape:", train_y.shape)

    print("test shape:", test_x.shape)
    print("test - label shape:", test_y.shape)

    train_label = pd.get_dummies(train_y)
    train_y = np.asarray(train_label)

    test_label = pd.get_dummies(test_y)
    test_y = np.asarray(test_label)
    
    print("Data is ready to be pushed to the black-box!")
    
    # determining configurations for the 
    config = Config(train_x, test_x, input_width)

    X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])
    Y = tf.placeholder(tf.float32, [None, config.n_classes])

    pred_Y = LSTM_Network(X, config)

    # Appending Loss
    l2 = config.lambda_loss_amount * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

    # Softmax loss
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=pred_Y)) + l2

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))
    
    print(""); print("Let's start trainning the data")
    
    training_epochs = 20
    loss_over_time = np.zeros(training_epochs)
    total_batches = train_x.shape[0] // config.batch_size
    b = 0
    
    start_time = time.time()
    # Launch the graph
    with tf.Session() as sess:
        # sess.run(init)
        tf.initialize_all_variables().run()
        # Keep training until reach max iterations
        # cost_history = np.empty(shape=[0],dtype=float)
        for epoch in range(training_epochs):
            cost_history = np.empty(shape=[0],dtype=float)
            for b in range(total_batches):
                offset = (b * config.batch_size) % (train_y.shape[0] - config.batch_size)
                batch_x = train_x[offset:(offset + config.batch_size), :, :]
                batch_y = train_y[offset:(offset + config.batch_size), :]

                # print "batch_x shape =",batch_x.shape
                # print "batch_y shape =",batch_y.shape

                _, c = sess.run([optimizer, cost],feed_dict={X: batch_x, Y : batch_y})
                cost_history = np.append(cost_history,c)
            loss_over_time[epoch] = np.mean(cost_history)
            print("Epoch: ",epoch," Training Loss: ",np.mean(cost_history)," Training Accuracy: ",sess.run(accuracy, feed_dict={X: train_x, Y: train_y}))
        
        end_time = time.time()
        print("");print("Training time: %s seconds" % (end_time - start_time) )
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_x, Y: test_y}))

        # MORE METRICS
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_x, Y: test_y}))
        
        # pred_Y is the result of the LSTM
        y_p = tf.argmax(pred_Y, 1)
        val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={X:test_x, Y:test_y})
        
        print("validation accuracy:", val_accuracy)
        y_true = np.argmax(test_y,1)
