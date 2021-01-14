# Prologue
Normally, it's difficult to plan out research. It's done when the research question has been answered, or the hypothesis been disproven.
I'm not quite there yet. So I'll discuss what I already did experiment with in more detail.


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

#### Learning

> Okay, so I understand that this neural network can differentiate vertical from horizontal. But you just defined some numbers and did a bit of math. How do computer's even come into play here? This just seems like glorified math!

Well, simple neural networks are just that! Glorified math. 
You're right that I wasn't playing completely fair here though. I skipped the learning part in machine learning. 

See, when you just have a bunch of fully connected layers, the behaviour of your network is completely defined by all those weights. That's where learning comes in. You have your model do what it's supposed to do, and based on how well it does it, you alter the weights, thus slowly teaching the network.(Kind of like mathematical logic)
There are many ways of doing this, but in my project, it was all handled automatically by tensorflow. So I won't go there in this document.

If you want to know more, I suggest taking a look at [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) and [evolutional networks](https://en.wikipedia.org/wiki/Neuroevolution) the [tensorflow youtube channel](https://www.youtube.com/channel/UC0rqucBdTuFTjJiefW5t-IQ) also has loads of information.

## Convolutional neural networks

Since I wanted to recognize gestures, which is just a form of image classification, convolutional networks were the best pick.
If you want to understand those, these four videos *[video 1](https://youtu.be/fNxaJsNG3-s) [video 2](https://www.youtube.com/watch?v=bemDFpNooA8) [video 3](https://www.youtube.com/watch?v=x_VrgWTKkiM&t=212s) [video 4](https://www.youtube.com/watch?v=u2TjZzNuly8)* are invaluable.

If you want to experiment with code samples without having to do any setup, this [online interactive environment](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/cnn.ipynb) is the way to go.

### What's the difference?

A convolutional network is very similar to the basic one explained above. The main difference is that you use convolutional layers, we'll come back to what those are later.

If you think about it, the previous kind of network could already do simple image classification. The examples with lines could in principle be treated as four pixels, right?
However, doing image classification that way imposes some big restrictions. If the subject in your image is slightly off-center for example, your network will have a **very** difficult time. That's only one of the issues you'd face though, imagine recognizing types of flowers or different organic shapes. They all have similarities, but they're definitely not going to nicely fit the same layout input-wise. 

> Hmm, that last sentence doesn't seem to be all negative. Sure, the input being inconsistent makes it harder. But similarities are a good thing right? Can't we go off of that?

Exactly! A convolutional neural network doesn't learn to match certain pixels to certain outputs, it extracts features and tries to match those!

> Cool, although, extracting features doesn't seem something our fully connected layers can do?

Fully connected layers are indeed not suited for this. Although we can use them to match the features to the output.

#### Convolutional layers

To extract features from our image, we'll use convolutional layers. Those layers iterate over the image and apply a convolution kernel. Those are simple matrices, usually 3x3. Applying them simply means matrix multiplication.

![Convolutional kernel being applied](https://mlnotebook.github.io/img/CNN/convSobel.gif)

> Okay, so we transform the data somehow, but how does all this work? 
> Where do the values for the kernels come from?
> How would this even extract any features?
> This affects the image resolution, right?

- In the gif above, the resolution is indeed affected. Although, this is easily countered by adding padding. There are other ways of dealing with it. But I won't go into these.
- The values of those kernels can be learned, just like the weights in previous example.
- Let me show you the magic

![Before and after convolution](https://timdettmers.com/wp-content/uploads/2015/03/convolution.png)
![Different convolution kernels in a gif](https://geekgirljoy.files.wordpress.com/2020/04/annkernelconvolutionexamples.gif)

Now, this process happens for every node in a convolutional layer. Remember the data *flowing* through the connections? That's a lot of data now...

#### Pooling

Since we want our models to actually be trainable on normal hardware, rather than supercomputers. We want to reduce our memory usage when possible. That's why, after using a convolutional layer, we'll usually put a layer that does some kind of pooling.

Pooling simply reduces the data, retaining the most prevalent parts. Kind of like how you can compress an image and still make out what's on there. Here's an example.
![max pooling example](https://media.geeksforgeeks.org/wp-content/uploads/20190721025744/Screenshot-2019-07-21-at-2.57.13-AM.png)

Here we simply take the maximum of a square, hence "max pooling" we can also do "average pooling" for example.

### Put together

So if we combine all of the above, we'll have an image as output. Our trained kernels will extract certain features from the image and the (max)pooling will scale down the features into digestible input for the fully connected layers at the end. Those will then match certain features to certain classes. And there we have it, image recognition!

![Complete convolutional network](https://pubs.rsc.org/image/article/2021/lc/d0lc01158d/d0lc01158d-f2_hi-res.gif)
