### Intro to Tensorflow
---
#### Convolutional Neural Network (CNN)

this example: simple_convolition.jpg is a good example.

Will take one pixel with its immediate neighbors, then define a filter and multiply the neighbors with the values in the filter, and sum them all up to define the next pixel's value.

The idea is that some convolutions will change the image in a way that some features in the image get emphasized.

**Pooling:**

The idea for pooling, is instead of taking one pixel, we can take a bigger area, like a 4x4 cut, then in this example(pooling_example.jpg) we take the darkest of each group, and convert them in a smaller convolution.

This way, we can have the more important pixels pop out in a short time, so that the model could easily recognize them.

To code this concept, we will keep the code from the last week, and keep the first layer flattened same size as the samples, one hidden layer, and the output layer to the same size as the class size. What we do now, is to add some layers on the top of this:
> model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu',
                            input_shape=(28,28,1)),
    ## the extra 1 is the color depth. since we are having greyscale images, we put 1 which is one byte)
    tf.keras.layers.MaxPooling2D(2,2),
    ## the line above means the pooling size will be 2 by 2, means for each 4 pixels, the darkest one will sirvive.                        
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')    
])

[here](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF) is a good source to explain different types of convolutions.

In fact, what we do as the convolutional layers, is to reduce the size of the images before feeding them into the dense layers.

**USEFUL METHOD:** model.summary() will show the whole journey of the image and convolutions.

We can add as many conv and pooling layers. We need to know that the convolution filter can't work for the pixels at the edges, because they have some missing neighbors. It means for the 3x3 filter, our input layer will be 26x26 instead of 28x28, because one pixel from each side will be deducted.

(model_summary_output.jpg)



