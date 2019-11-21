### Intro to Tensorflow
---
#### More advanced photo recognition with TF:

Tensorflow has an API called **imageGenerator.**

For using this feature, you can point out a folder of images, and this library will make a directory with separated folders for train, and test datasets, and inside each, have images for each object in seaprated folders.

(imageGenerator_folder_structure.jpg)

here is the code:
```
from tensorflow.keras.preprocessing import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory)
    train_dir,
    target_size=(300,300),
    batch_size=128,
    class_mode='binary')
```
flow from directory will load the images from the directory and its sub-directories.

- Be careful don't generate at the sub-directory, it doesn't work. we should always point to the main directory.

the name of the sub-directories will be the names of the labels.

target_size will take care of resizing all the images in runtime. no need to do it one by one.

it means you dont need to change the image sizes and it will resize them as they are loaded, not in the main source.

For validation generator we make the same thing:
```
test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(300,300),
    batch_size=128,
    class_mode='binary'
)
```
the difference is, we change the directory address.

Now to code the sample of horses, we will have three layers of convolutions and pools, the input shape will be (300,300,3) because we resized the images to 300x300 and 3 is because we use colors. one for Green, one for Red, and one for Blue.

for the output layer, we use __1 neuron with sigmoid activation (for two classes)__, it could also be __two neurons with softmax function.__ 

When we want to fit the model, because we dont use a dataset (we use streaming photo from imageGenerator), we dont use fit method, but we use fit_generator().

Then instead of the the dataset, we pass the train_generator we created earlier.

**Interesting:** softmax, can have multiple classes with the probability of each class as a percentage and the sum of them will be 1.

