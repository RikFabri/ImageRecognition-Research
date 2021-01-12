# Prologue
Normally, it's difficult to plan out research. It's done when the research question has been answered, or the hypothesis been disproven.
I'm not quite there yet. So I'll discuss the details of what I did experiment with in more detail.


# Research question
How can we apply machine learning to unity, to achieve image recognition in games?
### Goal
Create a simple unity game where the controls are based on the gestures you make in front of your webcam

# Research
The research question can be broken down in two main parts
> How does image recognition work?

> How to use neural networks in unity?

I decided to use [tensorflow](https://www.tensorflow.org/) and [python 3](https://www.python.org/) for the machine learning, since those are the most prevalent tools at the moment. 
For unity I stumbled upon [barracuda](https://docs.unity3d.com/Packages/com.unity.barracuda@1.2/manual/index.html), which is an experimental library to easily handle neural networks.

## Image recognition
### Machine learning
For the gesture recognition, I'm using a convolutional neural network. To understand what these are, we need to go back to the basics of machine learning first.

When people hear the terms AI/Machine learning/Neural networks, they often think about the same visualization.
![image from https://www.sitepoint.com/keras-digit-recognition-tutorial/](https://uploads.sitepoint.com/wp-content/uploads/2019/10/1571317553Artificial_neural_network-1024x914.png)

But what does it represent and how does it simulate intelligence?

#### The traditional neural network
To understand how this works, we'll construct a complete theoretical neural network from scratch. *(cool right?)* 

Our theoretical neural network, which we'll call Ben will obviously need a function. So let's train him to do something useful, like... recognizing lines!

We want to be able to tell whether or not a line is horizontal or vertical, based on these four squares as input

![Four 4x4 grids](/git_images/All_possibilities.png)

Now, Imagine that in the previous diagram, each circle on the first layer represents one of the squares. If a square is coloured, the circle has value one, if it's empty, it's zero. Every sphere is connected with all the spheres in the next layer *(called a fully connected layer)*. The lines between them have weights. The data from the input node flows through this line to the connected nodes, as it flows through the connection, it's multiplied by this weight. The node where the data ends up, takes the sum and often uses an activation function on that sum. We usually use the sigmoid function for this, which in essence just clamps the values between -1 and 1.

![Neural network sketch](/git_images/SimpleNet.png)

In the above drawing, The solid lines have a weight with value 0.5, the dotted lines have a value of 0. So that means that, the number in our first node will be:
```
A * 0.5 + 
B * 0.5 +
C * 0 +
D * 0
->
(A + B) / 2
```

This means that, the top node would be 1 if there's a horizontal line defined by A and B. Not coincidentally, a horizontal line on C and D would result in the second node being one
