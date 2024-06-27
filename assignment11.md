---
layout: default
title: Assignment 11
id: ass11
---


# Assignment 11: 
**Discussion: July 4th**  
**Deadline: July 3rd, 20:00**

This is the final assignment of the class. For once, we will not look at a new
type of generative model, but focus on a different modality, namely audio.
This is used in various tasks such as text-to-speech or music generation. To do
this properly would require a significant amount of setup and prior knowledge in
signal processing. As such, this should be taken as a very basic example showcasing
the wide applicability of deep generative models in different domains.


## The Data

Actually finding a dataset with desired properties is not simple:
- It should be small enough to be manageable to train in a one-week assignment.
- It should be large enough to make deep neural networks a sensible option.
- It should be high quality -- training on a low-quality dataset will result in
low-quality generations.
- For simplicity of modeling, it's preferable that all data points have the same
size (i.e. audio sequences of the same length).

We will settle for 
[Tensorflow Speech Commands](https://research.google/blog/launching-the-speech-commands-dataset/).
This does not fulfill the "high quality" requirement, but oh well. Since this is
not one of the nicely processed `keras.datasets`, we will have to some preprocessing ourselves.
We have two options:

First, use the `tensorflow_datasets` module and associated methods. An example can be found in
[this tutorial focused on classification](https://www.tensorflow.org/tutorials/audio/simple_audio).
Note that **I did not test this** as the `tensorflow_datasets` download have issues
on our group's servers. I used the second method instead.

Second option, follow these steps:
1. [Download the raw data](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz)
and unpack it. You may want to store this in Google Drive and mount the drive in
Colab (remember to set correct working directory etc.) 
-- else you will have to upload it into the runtime every time it resets.
2. In the course repository on Gitlab, you can find `data/process_tfsc.ipynb`.
This notebook walks through the process of going over the unpacked directory and
putting everything into a "TF Records" file format. You don't have to understand
the details of this format to use it. This format is read off disk in "chunks",
so the whole dataset doesn't need to be in memory at the same time (like with a
numpy array). It's also more efficient than loading many raw audio files one by one.
Make sure `base_path` is correctly set to the directory containing the uncompressed
dataset.
   - The script requires the `librosa` library to load audio. You will likely need
   to install this. It is only required for preprocessing, not for training etc.,
   so you don't have to keep installing it on Colab after the data has been processed.
   - If training later takes too long, you can reduce the `desired_sampling_rate` in preprocessing.
   This is set to `16000` by default. You could reduce it to `8000`; this reduces
   the dimensionality of the data, but also the audio quality. Even lower sampling
   rates can cause issues with audio playback, so this is not recommended. `4000` may still work,
   but likely the audio will sound no better than through a telephone.
   - Make sure to also adapt the sampling rate in your model code wherever needed.
   - The notebook was uploaded with output, so you can check what the expected
   output of each cell should be.
- Later in the model, you can create the dataset as shown in `assignment11_starter.ipynb`.
Basically, TFRecords is just pure uninterpreted bytes, and we have to tell Tensorflow
how to parse those bytes back into tensors


## The Model

You can find a starter notebook on Gitlab. This does not include the model; only
preparation of the dataset and example code on how to play audio (so you can later
listen to your generations). You can now, in principle, train any model we have
previously implemented on this dataset (diffusion models work well). The only difference is that the data now
has shape `(16000, 1)` -- 16000 time steps and one channel (**you need to add the channel
axis!**).  
Note: If you changed the sampling rate in preprocessing, be sure to also change
it in the model! Then of course, the data shapes will be different.
- You can use networks with 1D-convolutions (`layers.Conv1D`) and similar
1D analogues for pooling, upsampling, etc. Aside from that, the architectures can
stay exactly the same!
- You likely want to use larger filters and more aggressive striding than for images.
E.g. you might try a filter size of 7 and strides of 4.
- If using the previous diffusion models as a basis (like `assignment10`), 
there are a few more changes needed as some parts of the code broadcast lower-dimensional
tensors to the dimensionality of the data, which is 4D for images but only 3D for
audio. In the assignment 10 upload, these points are marked with capital `# WARNING`
comments. Note that I may have forgotten to mark some parts -- if you run into
trouble with dimensions, feel free to ask for help!
- Of course, if you decide to use another framework like VAEs or GANs, the above does
not apply.
