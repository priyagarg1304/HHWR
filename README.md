# HHWR
Hindi Hand Writing Recognition
This project recognizes Devanagri characters in realtime taking imput from the laptops's webcam.
The model is built using a Convolutional Neural Networks.
The model is trained on 70,000 images of Devanangri script character images of 32x32 size.
The neural network consists of two sets of Conv2D layer+MaxPooling layers.
Image taken from the webcam is converted from RGB to HSV and as the symbols are wriiten from a blue color only that portion of image is taken. Various blurs are applied to it like Median and Gaussian Blur and Contours(image boundaries) are detected.
The Contours with areas greater than a threshold are taken and drawn on a blackboard screen which then becomes the image input for the CNN trained previously.



