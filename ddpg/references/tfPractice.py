import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])
product = tf.matmul(matrix1, matrix2)

# method 1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)

#--------------------------------------------------------------------
state = tf.Variable(0, name = 'counter')
one = tf.constant(1)

new_value  = tf.add(state, one)
update = tf.assign(state, new_value)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

#--------------------------------------------------------------------

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))


def addLayer(inputs, in_size, out_size, activationFunction = None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    outputs = activationFunction(Wx_plus_b) if activationFunction is not None else Wx_plus_b
    return outputs

def main():
    x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
    y_data = np.square(x_data) - 0.5 + noise

    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])

    l1 = addLayer(xs, 1, 10, activationFunction=tf.nn.relu)
    prediction = addLayer(l1, 10, 1, activationFunction=None)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    plt.ion()
    plt.show()

    for i in range(1000):
        # training
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            # to visualize the result and improvement
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            # plot the prediction
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            plt.pause(0.1)


# %%
class Coursera:
    def compute_cost(Z3, Y):
        logits = tf.transpose(Z3)
        labels = tf.transpose(Y)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        return cost

    def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
              num_epochs=1500, minibatch_size=32, print_cost=True):
        ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
        tf.set_random_seed(1)  # to keep consistent results
        seed = 3  # to keep consistent results
        (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
        n_y = Y_train.shape[0]  # n_y : output size
        costs = []  # To keep track of the cost

        X, Y = create_placeholders(n_x, n_y)
        parameters = initialize_parameters()

        Z3 = forward_propagation(X, parameters)
        cost = compute_cost(Z3, Y)

        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Initialize all the variables
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(num_epochs):
                epoch_cost = 0.
                num_minibatches = int(m / minibatch_size)
                seed = seed + 1
                minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

                for minibatch in minibatches:
                    (minibatch_X, minibatch_Y) = minibatch
                    _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                    epoch_cost += minibatch_cost / num_minibatches

                if print_cost == True and epoch % 100 == 0:
                    print("Cost after epoch %i: %f" % (epoch, epoch_cost))
                if print_cost == True and epoch % 5 == 0:
                    costs.append(epoch_cost)

            parameters = sess.run(parameters)
            print("Parameters have been trained!")

            correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
            print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

            return parameters
