
import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
import time
import scipy.ndimage

class Neural_Network:
    """
    inodes = number of node in input layer
    hnodes = number of node in hidden layer
    onodes = number of node in output layer
    wih = weights matrix of input and hidden layer
    who = weights matrix of hidden and output layer
    lr = learning rate
    activation_function = sigmoid function

    # how it works:
    # step 1: get the list input, calculate output array
    # step 2: taking the calculated output, comparing with desired output, and
            using the difference to guid the updating of the network weights
    """
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate

        # weights inside the arrays are w_i_j where link is from node i to node j
        # in the next layer
        # w11 w21
        # w12 w22
        # w13 w23 etc
        # the initial value is in range -0.5, +0.5 (normally -1,1)
        self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)
        #print(self.wih)
        #print(self.who)

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)



    def train(self, input_list, target_list):
        """
        train the neural network
        :return:
        """
        # convert inputs list to 2d array
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T

        #### 1. feed forward the signals
        # calculate the signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the output of hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals into output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outpus = self.activation_function(final_inputs)


        #### 2. backpropagate the errors
        # output layer error = (target - actual)
        output_erros = targets - final_outpus
        # hidden layer error is the output_errors, split by the weights, recombined at hidden nodes
        hidden_erros = numpy.dot(self.who.T, output_erros)


        #### 3. update the weights
        # update the weights for the links between the hidden and output layer
        # nn learn from the errors
        # formula Wjk += lr * Ek * sigmoid(Ok)*(1-sigmoid(Ok)) * OjT
        self.who += self.lr * numpy.dot((output_erros * final_outpus*(1 - final_outpus)), numpy.transpose(hidden_outputs))
        # update the weights for the links between the input and hidden layer
        self.wih += self.lr * numpy.dot((hidden_erros * hidden_outputs*(1 - hidden_outputs)), numpy.transpose(inputs))


    def query(self, input_list):
        """
        query the neural network
        just to see with the list of input, what do we get from the nn
        :return:
        """

        # convert inputs list to 2d array
        inputs = numpy.array(input_list, ndmin=2).T
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


if __name__ == '__main__':
    input_nodes = 28*28
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.1

    # create the nn
    nn = Neural_Network(input_nodes, hidden_nodes, output_nodes, learning_rate)
    try:
        file = open('../datasets/mnist_train.csv')
        data_list = file.readlines()
        file.close()
        # go through the training data set
        # take out an image
        # give it to the network
        ### 1. Train the network with training data
        print("Start training {} images".format(len(data_list)))
        start = time.time()
        count = 0
        for record in data_list:
            all_values = record.split(',')
            # we don't need the first element, because it's the target value
            # now convert the string array to float array, and scale down to range(0.01, 0.99)
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # the target is an array length 10, which looks like:
            # [0.01, 0.01, ..., 0.99, 0.01]
            # if 0.99 is the n value in the array, then that's the number we're looking for
            targets = numpy.zeros(nn.onodes) + 0.01
            targets[int(all_values[0])] = 0.99
            nn.train(inputs, targets)

            ## create rotated variations
            # rotated anticlockwise by x degrees
            inputs_plusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), 10, cval=0.01, order=1,
                                                                      reshape=False)
            nn.train(inputs_plusx_img.reshape(784), targets)
            # rotated clockwise by x degrees
            inputs_minusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), -10, cval=0.01, order=1,
                                                                       reshape=False)
            nn.train(inputs_minusx_img.reshape(784), targets)
            count += 1
            if count % 1000 == 0:
                print("{:.2f}%".format(count / float(60000) * 100))
        now = time.time()
        print("Finish training\nTook {} s".format(now - start))

        ### 2.Testing
        # load testing data
        file = open('../datasets/mnist_test.csv')
        test_data_list = file.readlines()
        file.close()
        print("Start testing", len(test_data_list), "images")
        start = time.time()
        count_correct = 0
        for record in test_data_list:
            all_values = record.split(',')
            correct_label = int(all_values[0])
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            outputs = nn.query(inputs)
            # the index of the highest value coressponds to the label
            label = numpy.argmax(outputs)
            if label == correct_label:
                count_correct += 1
        print("Finish testing\nTook ", time.time() - start,
              "s\nAccuracy: {:.2f}%".format(count_correct / len(test_data_list) * 100))
        print nn.query([0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    except Exception as e:
        print(e)
