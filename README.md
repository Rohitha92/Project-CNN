# Project-CNN
This Project is Convolutional Neural Network based Image Classification, aiming at classifying 3 classes - glass bottle, plastic bottle and tin/aluminium bottle. <\br>
The dataset consists of 1800 images (600 in each class), 300 validation images and 300 test images. <\br>
The data is stored in .mat file
## Requirements
   - Tensorflow 0.10, numpy, scipy, sklearn
## Script Initilatization
   - Run Train.py for training the network.
   - Run test_images.py for testing the trained network on new images in a folder. Sample new images are provided.
   - Change paths in the program accordingly before executing
   - Checkpoints folder consists of trained model
### Dataset Credits:
    - Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya
    Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution) ImageNet Large Scale Visual Recognition
    Challenge. IJCV, 2015
    - Gary Thung, Dataset of garbage images, https://github.com/garythung/trashnet
