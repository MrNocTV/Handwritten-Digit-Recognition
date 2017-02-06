
from Tkinter import *
from PIL import ImageGrab, Image, ImageTk
import NeuralNetwork
import DigitDetection
import time
import numpy
import scipy.misc
import scipy.ndimage
import cv2
import os


b1 = "up"
xold, yold = None, None
drawing_canvas, root, image_canvas, showing_result_canvas = None, None, None, None
input_nodes = 28 * 28
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.1
nn = NeuralNetwork.Neural_Network(input_nodes, hidden_nodes, output_nodes, learning_rate)
label = None
targets = None

def main():
    global drawing_canvas, root, image_canvas, showing_result_canvas, result_text
    root = Tk()
    root.title("Digits Recognition")  # set title
    root.resizable(0, 0)
    drawing_canvas = Canvas(root, width=28 * 10, height=28 * 10)
    drawing_canvas.grid(row=0, column=0, padx=5, sticky='nswe')
    drawing_canvas.bind("<Motion>", motion)
    drawing_canvas.bind("<ButtonPress-1>", b1down)
    drawing_canvas.bind("<ButtonRelease-1>", b1up)
    drawing_canvas.bind("<Button-3>",
                        lambda x: drawing_canvas.delete('all'))  # clear canvas when user click right button
    drawing_canvas.create_rectangle((10, 10, 270, 270))
    image_canvas = Canvas(root, width=28 * 10, height=28 * 10, bg='blue')
    image_canvas.grid(row=0, column=1, padx=5, pady=5)
    image_canvas.config(highlightthickness=1)
    showing_result_canvas = Canvas(root, height=60, background='greenyellow')
    showing_result_canvas.create_text(20, 40, anchor=W, font=("Purisa", 20),
                                      text="Draw any digit you want")
    showing_result_canvas.grid(row=1, column=0, columnspan=2, sticky=E+W)

    root.mainloop()  # start the event loop


def b1down(event):
    global b1
    b1 = "down"  # you only want to draw when the button is down
    # because "Motion" events happen -all the time-


def b1up(event):
    global b1, xold, yold, drawing_canvas, root, image_canvas
    b1 = "up"
    xold = None  # reset the line when you let go of the button
    yold = None
    # save fullsize image
    if not os.path.isdir('../images'):
        os.mkdir('../images')
    ImageGrab.grab().crop((root.winfo_x()+20, root.winfo_y() + 30, 350, 350)).save("../images/save.png")
    img = Image.open("../images/save.png")
    img_array = cv2.imread("../images/save.png")
    img_array = DigitDetection.detect(img_array)
    img = Image.fromarray(img_array, 'RGB')

    # resize the image to fit the neural network
    img = img.resize((28, 28), Image.ANTIALIAS)  # best down-sizing filter
    img.save("../images/save1.png")  # this is used for recognizing
    img.save("../images/save.gif")  # this is used for displaying

    # display image after being processes
    img1 = Image.open("../images/save.gif")
    img1 = img.resize((28 * 9 + 20, 28 * 9 + 20))
    img1.save("../images/save_scaled.gif")
    photo = PhotoImage(file = "../images/save_scaled.gif")
    image_canvas.create_image(5, 5, image = photo, anchor = 'nw')
    label = Label(image=photo)
    label.image = photo  # keep a reference!

    # recognize
    recognize_from_test("../images/save1.png")

def motion(event):
    if b1 == "down":
        global xold, yold, image, draw
        if xold is not None and yold is not None:
            # here's where you draw it. smooth. neat.
            event.widget.create_line(xold, yold, event.x, event.y, smooth=TRUE, width=11, capstyle=ROUND, joinstyle=ROUND)

        xold = event.x  # update x
        yold = event.y  # update y


def start_training():  # train from mnist dataset
    # load data
    global nn
    try:
        file = open('../datasets/mnist_train.csv')
        data_list = file.readlines()
        file.close()
        # go through the training data set
        # take out an image
        # give it to the network
        ### 1. Train the network with training data
        print("Start training {} images".format(len(data_list)*3))
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
        who = nn.who
        wih = nn.wih
        print(who)
        print(wih)
        if not os.path.isdir('../weight_matrices'):
            os.mkdir('../weight_matrices')
        numpy.save('../weight_matrices/who.out', who)
        numpy.save('../weight_matrices/wih.out', wih)
        print('Saved Successfully')

    except Exception as e:
        print(e)

def recognize_10000_samples_data():
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
          "s\nAccuracy: {:.2f}%".format(float(count_correct) / len(test_data_list) * 100))

def recognize_from_test(path):
    global showing_result_canvas
    # load image in gray scale
    img_array = scipy.misc.imread(path, flatten=True)
    print img_array
    img_data = 255.0 - img_array.reshape(28 * 28)  # convert to mnist dataset
    # scale to fix the sigmoid function
    img_data = (img_data / 255.0 * 0.99) + 0.01
    outputs = nn.query(img_data)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    showing_result_canvas.delete('all')
    showing_result_canvas.create_text(20, 40, anchor=W, font="Purisa",
                                      text="Your number is {},".format(label) + " is that correct?")


if __name__ == "__main__":
    if not (os.path.isfile('../weight_matrices/who.out.npy') and os.path.isfile('../weight_matrices/wih.out.npy')):
        start_training()
    else:
        nn.who = numpy.load('../weight_matrices/who.out.npy')
        nn.wih = numpy.load('../weight_matrices/wih.out.npy')
    recognize_10000_samples_data()
    main()
