# Data_Augmentation

Data Augmentation is the process of randomly transforming image to produce new synthetic images.

Some of the useful transformation are:-
-Resize

-Flipping

-Lighting

-Color Distort

-Shifting

-Expand

-Crop

-Auto Augment

In keras, we have a module ImageDataGenerator for generating the augmented images. ImageDataGenerator can be used on single images as well as multiple images all at once in batches. ImageDataGenerator can be used for single class or multi-class dataset augmentation.

Here is a small example of Image augmentation for single image, multiple images(in batches) and multiple class images.
