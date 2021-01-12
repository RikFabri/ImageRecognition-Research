# Prologue
Normally, it's difficult to plan out research. It's done when the research question has been answered, or the hypothesis been disproven.
I'm not quite there yet. So I'll discuss the details of what I did experiment with in more detail.


# Research question
How can we apply machine learning to unity, to achieve image recognition in games?
### Goal
Create a simple unity game where the controls are based on the gestures you make in front of the webcam

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

### The traditional neural network
To understand how this works, we'll construct a complete theoretical neural network from scratch. *(cool right?)* 

Our theoretical neural network, which we'll call Ben will obviously need a function. So let's train him to do something useful, like... recognizing lines!

