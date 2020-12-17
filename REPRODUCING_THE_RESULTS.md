**

## Recreating the results
 
 **Training the pix2pix model on the cityscapes dataset**

 1. Go to https://www.cityscapes-dataset.com/login/ and download the gtFine_trainvaltest dataset
 2. use the python script prepare_cityscapes_dataset.py in the folder datasets according to its --help instructions to prepare the dataset to the required format. 


 **Training the pix2pix model on the cityscapes dataset**

 1. Using pip install the requirements in requirements.txt with pip install -r requirements.txt
 2.  follow the instructions in conda-env.txt to install the rest of the required packages using anaconda (you can also use the import button in anaconda navigator and select conda-env.txt)
 3. for training: in the command line run `python train.py --n_epochs 100 --n_epochs_decay 100 --dataroot ./datasets/cityscapes --name cityscapes_pix2pix --model pix2pix --direction BtoA`

the dataset is contained already in the repo

 4. for testing run `python test.py --dataroot ./datasets/cityscapes --direction BtoA --model pix2pix --name cityscapes_pix2pix`





## Results

**Cityscapes Dataset**

The Cityscapes Dataset consists of a few thousand images of street scenes with semantic segmentaton as annotation. It was aquired in many european cities, including Zurich. The images have partially been annotated by hand on a per-pixel basis. The rest of the images has croase annotations. The usecase lies mainly in autonomous driving, as in this task the cars computer has to recognise where the street is and what is a sidewalk for example.
![Example](\imgs\1.jpg)

**Training pix2pix on cityscapes with L1 loss**
L1 loss creates already fairly raelistic looking images, eventhough they seem to have slightly less details than with perceptual loss. The images appeared to be a little bit mor washed out and averaged by the network. Also they are a bis mir gray sometimes. This is, because the L1 loss is also small if the network makes the pixel value of the output image match the value of the average pixel at that position for the given class. This happens to be gray often, so the network learns to use gray a lot. t the end of the Training one can see that the L1 loss is still giong down relativeley fast, wich might suggest that the network is not overfitting a lot. If the network would be overfittting at the end of the training, It would probably have more similar colors in its output images compared to the ground truth.
![Example](\imgs\2.jpg)

**Training pix2pix on cityscapes with perceptual loss**
With prceptual loss the images looked even more realistic and detailed. However it also took about 5 times longer to train the network. This it due to the fact that the perceptual loss, compared to L1 loss is quite a heavy computation. With the perceptual loss the learning objective is less focused on the specific color of every pixel of the image. This is good since for example for a car in the image, almost any color could be correct. Here the L 1 Loss would "punish" the networkfor the wrong color. Hoever only from the Input image there is no way of knowing that color certain objects like cars have in any specific case.
![Example](\imgs\3.jpg)

## Graphical user interface

The GUI is intended to show the capabilities of the networks that we have trained so far, and also the differences between the different models. There are different buttons on the top that allow the user to select a color and draw in the image. By the click on the edit button the image is then processed and the output of the network is displayed. Those also had to be changed to fit the new labels of our dataset.
We attempted toadapt the gui that was provided in the Git repo to our own model, however

**GUI for testing the model:**
 1. Execute the file CelebAMask-HQ-master\MaskGAN_demo\demo.py using the command `python CelebAMask-HQ-master\MaskGAN_demo\demo.py` this will start the GUI
 2. in the GUI select a input image form the \MaskGAN_demo\samples folder and the corresponding labels. For example 0.jpg and 0.png. 

To run the gui some debugging was needed: When running it on windows you the paths all had to be changed to absolute paths with double backslashes instead of backslashes in order for it to work.

This is an example of how you can edit the picture and directly see the effect.
In this example the mask and image did not match puroposely. We see that the neural network takes the style from the input image instead of only using the semantic labels. 
Therefore, the structure of the GUI software and its model made ist quite hard to replace the model with one that we had trained ourselves.
 ![Example](\imgs\4.jpg)