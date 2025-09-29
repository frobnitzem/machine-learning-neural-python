---
title: Next Steps
teaching: 20
exercises: 20
---

::::::::::::::::::::::::::::::::::::::: objectives

- Connect this tutorial to the larger field of AI/ML.
- Get a sense of other ML frameworks and packages.
- Know where to look to for examples of other datasets / methods / architectures.

::::::::::::::::::::::::::::::::::::::::::::::::::


:::::::::::::::::::::::::::::::::::::::: questions

- How do classifier, autoencoder, segmentation, and generative architectures differ?
- When should I use a cross-entropy loss vs an MSE loss?
- What is the "best" deployment path for datasets and models?

::::::::::::::::::::::::::::::::::::::::::::::::::


## Model types

All neural networks are "trainable functions." However, what gets input
to those functions, and how the output is used, are up to the modeler.
This tutorial has covered a classifier, whose input is an image or some
text, and which outputs predicted categories.  Usually, this means the
output is a vector of numbers -- one for each category.  The highest
number "wins" and becomes the predicted classification for that input.

### Classifiers

For classifiers like this, it's possible to treat the numbers as probabilities
for each category assignment.  Thus, a good loss function is the cross-entropy,

$$
L(x, p_c; \theta) = -\sum_c p_c \log (f_c(x; \theta)/p_c)
$$

where $p_c$ is the "known" probability that sample $x$ has category $c$,
and $\theta$ are all trainable parameters in the function, $f$.
This measures how well the prediction function, $f(x)$, matches the
known probability distribution.  The loss function is minimized, so
lower should be "better" agreement.

In the case there is only one label, $p_c$ is zero except for one particular
$c$.  Then the loss function is simply,
$$
L(x, c; \theta) = -\log f_c(x; \theta)
$$

### Image Segmentation

Image segmentation models are like classifiers, but predict output categories
for every pixel in the image.


### Autoencoders

An autoencoder seeks to train a network whose output reproduces its input.
The loss function should be meaningful to the type of input used.  For example,
image autoencoders typically use mean-squared-error (MSE).  MSE takes the mean
of the square difference over all pixels (predicted vector dimensions).

It may be counterintuitive to train a network to predict its
input (which is already known).  However, autoencoders have the advantage
that they do not need training labels (e.g. category assignments).
They also show to what degree a given network architecture can "learn"
to represent its input.  Since autoencoders almost always have a
"bottleneck" layer with few parameters, the data at this layer serves
as a compressed representation for the set of inputs.

Most techniques with autoencoders revolve around the representation in
"latent space". This is the distribution of points seen at the bottleneck.
For example, randomly picking some point in latent space and then running
the rest of the network is called decoding.  Decoding usually creates
an output that looks like something from its input set.

### Generative Models

Generative AI models are like autoencoders, but usually contain
only the decoder part of the architecture.
In place of an encoder for building the latent space,
the distribution over latent space is prescribed.

For generative image models, a typical example is
to set the distribution of latent space points to a Gaussian.
The loss function for a given sample is like

$$
L(x) = \sum_l f_l(x)^2 / sigma^2_l
$$

where $f_l(x)$ is the network's representation of $x$ at layer $l$,
and $\sigma_l$ is the noise level at layer $l$.

Text generation also uses generative models (most recently
transformer architectures).  Different from images, these models
are more like categorization in that they start from a segment of text
and produce, as output, a set of probabilities for the next token.

### Combination Models

Model types can be combined by concatenating two vectors together,
making a "combined latent vector".  The first few steps of
a good NN model take the input to a "feature vector".  Given good
feature vectors for two different types of input (for example an image
and descriptive sentence), combined methods can be used to do things like
better segment images, answer questions about images,
compare two images/sentences, etc.


:::::::::::::::::::::::::::::::::::::::: keypoints

- Neural networks model general functions.
- AI/ML methods differ in how the network inputs are constructed, and how their outputs are interpreted.
- Inputs are usually mapped to "features", which are compressed summaries of
  the data.
- Many network types make use of a final set of fully connected networks (multi-layer perceptrons).

::::::::::::::::::::::::::::::::::::::::::::::::::


## Next Steps - Links

Modernize with pre-trained models:

 - <https://codezup.com/hugging-face-transformers-medical-imaging/> 

[Scikit-Learn](https://scikit-learn.org/stable/auto_examples/index.html) & [Scikit-Image](https://scikit-image.org/docs/stable/auto_examples/) 

 - lots of background info. and "pre-baked" models/data here,
   but little ability to customize model internals 

[PyTorch](https://pytorch.org/) 

 - highly used python AI/ML framework with strong developer
   community 
 - torch.Tensor similar to numpy.array, but with auto-differentiation
   and GPU support 

[HuggingFace](https://huggingface.co/)

 - A "Facebook for data and ML" 
 - provides key libraries for loading datasets and models from the cloud 
 - guiding community standards in data 

[MLFlow](https://mlflow.org/docs/latest/ml/tutorials-and-examples/)

 - a cloud-like tracker for your datasets and models 
 - highly recommended for model save/load functionality 
 - data is hosted locally on your computer 
 - can scale to local server-based deployment 

Specifically, `mlflow.tensorflow.autolog()` for [autologging](https://mlflow.org/docs/2.21.3/tracking/autolog#autolog-keras) in the [example](https://github.com/mlflow/mlflow/blob/master/examples/keras/train.py)

Pneumonia classifier in a web app:

 - [project description](https://www.matiasbrandt.com/computer-vision/building-an-image-classifier-scikit) 
 - [published dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2) 
 - [web interface](https://mbm-pneumonia-classifier.streamlit.app/) 
 - [source code](https://github.com/brandtbrandtbrandt/pneumonia-classifier-streamlit-app) 
 - Dependencies: pillow (import PIL), tensorflow/keras, streamlit 

