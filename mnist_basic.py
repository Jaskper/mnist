import numpy
from mnist import MNIST

#activations should be non-negative (adjusted with ReLU)
#weights & biases have no bounds

#adjust outputs using softmax
#uses cross-entropy to calculate cost function

weights_learning_rate = 0.01
bias_learning_rate = 0.02

mndata = MNIST('./mnist')
images, labels = mndata.load_training()

#applies the sigmoid function to an array of values and returns the altered array
def sigmoid(data_array):
    return_array = []
    for value in data_array:
        return_array.append(1.0/(1.0+numpy.exp(-value)))

    return return_array

#returns the array of 0s and 1s cooresponding to the goal output for a given image label
def single_image_answer(image_label):
    answers_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    correct_answer = labels[image_label]
    answers_array[correct_answer] = 1
    return answers_array

#returns the error of an output by comparing it to the goal array
def output_error(output_layer, image_label):
    correct_answers = single_image_answer(image_label)
    overall_error = 0
    for index in range(10):
        current_error = (correct_answers[index]-output_layer[index])**2
        overall_error += current_error
    return overall_error/2

#between input layer & first hidden layer
first_weights = numpy.random.normal(0,1,(784,20))
first_bias = numpy.zeros((1,20))

#between first hidden layer & second hidden layer
second_weights = numpy.random.normal(0,1,(20,20))

second_bias = numpy.zeros((1,20))

#between second hidden layer & output layer
third_weights = numpy.random.normal(0,1,(20,10))
third_bias = numpy.zeros((1,10))

input_layer = numpy.zeros((1,784))
first_hidden_layer = numpy.zeros((1,20))
second_hidden_layer = numpy.zeros((1,20))
output_layer = numpy.zeros((1,10))

for i in range(5):
    for current in range(53001):

        current_image = current

        ###########FORWARDS PASS#################

        #adjusts input images number values from 0-255 to 0-1
        adjusted_image = []
        for value in images[current_image]:
            adjusted_image.append(value/255)

        #input_layer = numpy.matrix(adjusted_image)
        input_layer = adjusted_image

        #first hidden layer - size 20
        first_hidden_layer = sigmoid(numpy.dot(input_layer, first_weights) - first_bias)

        #second hidden layer - size 20
        second_hidden_layer = sigmoid(numpy.dot(first_hidden_layer, second_weights) - second_bias)

        #output layer
        output_layer = sigmoid(numpy.dot(second_hidden_layer, third_weights) - third_bias)

        trial_error = output_error(output_layer[0], current_image)

        ###########BACKWARDS PASS##################

        third_weights_gradient = numpy.zeros((20,10)) #should be matrix of size 20 by 10
        third_bias_gradient = numpy.zeros((1,10)) #should be matrix of size 1 by 10

        second_weights_gradient = numpy.zeros((20,20)) #should be matrix of size 20 by 20
        second_bias_gradient = numpy.zeros((10,20)) #should be matrix of size 1 by 20

        first_weights_gradient = numpy.zeros((784,20)) #should be matrix of size 784 by 20
        first_bias_gradient = numpy.zeros((10,20)) #should be matrix of size 1 by 20

        error_over_output_values = []

        #goes through all the outputs cells
        for output_cell_index in range(10):
            output_cell_value = output_layer[0][output_cell_index]
            target_value = single_image_answer(current_image)[output_cell_index]

            current_bias_gradient = (output_cell_value - target_value) * (output_cell_value * (1 - output_cell_value))
            third_bias_gradient[0][output_cell_index] = current_bias_gradient
            error_over_output_values.append(current_bias_gradient)

            #goes through all the weights on the third weights matrix
            for weight_number in range(20):
                current_weight_gradient = (output_cell_value - target_value) * (output_cell_value * (1 - output_cell_value)) * second_hidden_layer[0][weight_number]
                third_weights_gradient[weight_number][output_cell_index] = current_weight_gradient

        error_over_second_hidden_values = []

        #goes through all the second hidden layer cells
        for second_hidden_cell_index in range(20):
            cell_value = second_hidden_layer[0][second_hidden_cell_index]

            error_over_second_hidden = 0
            for output_cell_index in range(10):
                error_over_second_hidden += error_over_output_values[output_cell_index] * third_weights[second_hidden_cell_index][output_cell_index]
            error_over_second_hidden_values.append(error_over_second_hidden)

            current_bias_gradient = error_over_second_hidden * (cell_value * (1 - cell_value))
            second_bias_gradient[0][second_hidden_cell_index] = current_bias_gradient

            for weight_number in range(20):
                current_weight_gradient = error_over_second_hidden * (cell_value * (1 - cell_value)) * first_hidden_layer[0][weight_number]
                second_weights_gradient[weight_number][second_hidden_cell_index] = current_weight_gradient

        for first_hidden_cell_index in range(20):
            cell_value = first_hidden_layer[0][first_hidden_cell_index]

            error_over_first_hidden = 0
            for second_cell_index in range(20):
                error_over_first_hidden += error_over_second_hidden_values[second_cell_index] * second_weights[first_hidden_cell_index][second_cell_index]

            current_bias_gradient = error_over_first_hidden * (cell_value * (1 - cell_value))
            first_bias_gradient[0][first_hidden_cell_index] = current_bias_gradient

            for weight_number in range(784):
                current_weight_gradient = error_over_first_hidden * (cell_value * (1 - cell_value)) * input_layer[weight_number]
                first_weights_gradient[weight_number][first_hidden_cell_index] = current_weight_gradient

        third_weights = third_weights - weights_learning_rate * third_weights_gradient
        third_bias = third_bias - bias_learning_rate * third_bias_gradient
        second_weights = second_weights - weights_learning_rate * second_weights_gradient
        second_bias = second_bias - bias_learning_rate * second_bias_gradient
        first_weights = first_weights - weights_learning_rate * first_weights_gradient
        first_bias = first_bias - bias_learning_rate * first_bias_gradient

        print(trial_error)
#TESTER

total_correct = 0
for current in range(500):
    current_image = current + 54000

    ###########FORWARDS PASS#################

    #adjusts input images number values from 0-255 to 0-1
    adjusted_image = []
    for value in images[current_image]:
        adjusted_image.append(value/255)

    #input_layer = numpy.matrix(adjusted_image)
    input_layer = adjusted_image

    #first hidden layer - size 20
    first_hidden_layer = sigmoid(numpy.dot(input_layer,first_weights) - first_bias)

    #second hidden layer - size 20
    second_hidden_layer = sigmoid(numpy.dot(first_hidden_layer, second_weights) - second_bias)

    #output layer
    output_layer = sigmoid(numpy.dot(second_hidden_layer, third_weights) - third_bias)

    current_max = numpy.argmax(output_layer[0])
    correct = labels[current_image]

    if current_max == correct:
        total_correct += 1

    print("guess " + str(current_max) + " for " + str(correct))
print("percent accuracy: " + str(total_correct/500))
