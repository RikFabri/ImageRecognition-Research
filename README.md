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

In the above drawing, we'll define the weights so that the top two *(solid circles)* nodes look for horizontal lines. In essence, this means we want them to be one if a line is horizontal, or less if the line is vertical.
To achieve this, we'll give the solid lines a weight of 0.5 and the dotted lines a weight of 0. 

We'll do the resulting math of the top node, while keeping in mind that every line multiplies the data with it's weight and the nodes take the sum of all incoming lines.
So that means that, the number in our first node will be:
```
A * 0.5 + 
B * 0.5 +
C * 0 +
D * 0
->
(A + B) / 2
```

This does indeed mean that, the top node would be 1 if there's a horizontal line defined by A and B. Not coincidentally, a horizontal line on C and D would result in the second node being one

We can use similar logic to have the bottom two nodes look for vertical lines. We'd just need to assign the 0.5 to the right connections per node.

> Alright, I get it. But we have our answer right? the top two nodes mean horizontal, the bottom two are vertical.
> Then why are there still so many nodes left?

Well, horizontal and vertical are only two options. I thought I'd simplify the network's output to show you how it can do many different kinds of logical operations. The remaining nodes serve nothing but reducing the output into one value, which will be negative for vertical lines and positive for horizontal ones.

Let me explain: We gave all connections to the top node a weight of one and all connections to the bottom node minus one. This means that if the top node has a bigger value than the bottom one, the result will be positive. The bottom node however, negated all the input. So if the absolute value of the bottom nodes is bigger (thus, meaning vertical) the output node will be negative instead. And just like that, we have a very easy to work with output.
