# Digit-masking-in-an-Image
The task is to mask the digits in an image, having a string of characters.

Problem Statement: 

 GitLab Link

Given an input image with string of specified length present on it, generate an output that will be the same image but a portion of the string(digits) masked. String can be anywhere in the image.

String will be of specified length and will have a sequence of digits present in it. Masking means, fill the digit portion of the string with colour of your choice. 

Here's what I thought:
	• Detect ROI of individual characters
	• Extract each character as an independent image
	• Pass it through a CNN model and recognize the character
	•  If its a digit, then mask it
	• Paste all the extracted image back to the original image 

	Detecting ROIs :
		○ This is so frustrating! Loading the net, creating 'blob'...JESUS.
		○  Thought I was making progress using pure OpenCV classes and function but all the effort for no results (well minimal results..).
	
	Thresholding an image:

	 If pixel value is greater than a threshold value, it is assigned one value (may be white), else it is assigned another value (may be black).
	
				
	
	Dilation :

	 A pixel element is ‘1’ if at-least one pixel under the kernel is ‘1’. So it increases the white region in the image or size of foreground object increases.
	
	

## Finally getting some analysis done. But passing the entire image makes it too blurry or depreciated to be of any use.

-> First need to crop the image

I got it done by just selecting the rows and columns that has values other then a constant value (which is the greyness of the background).

To crop the image with just Text

							
 
 


								
								
								
OK. It works for "MOST" of the images, but not for image1. Moving on to the next step.

->To split the cropped image into multiple images each having a single character.

	• A probable Dataset for training CNN

	• Another One

	1. Depreciated the image using Threshold.
	2. Using Contours to detect the individual characters. The output of the contours is actually a list of arrays (x, y) coordinates of boundary points of each object. 
	3. Tested by plotting the detect contours back on the image..it works.
	4. Plotting a bounding rectangle around each contour.

		
		'Y' and 'T' are not being differentiated. Trying to erode the image a bit.
		
	5. Getting a pretty decent output now, but the number of images I am getting is greater than the number of characters in the image, due to the presence of inner contours.
	6. Learning to use hierarchy to remove inner contours.
		○ That was simple, just changed the retrieval mode from RETR_TREE to RETR_EXTERNAL.
		
		Returns exactly 12 images. ☺
		
	• NOTE: It is having trouble properly detecting letters but not with digits. So I am going ahead with this faulty detection since I only have to work with digits.
	• Also I changed from thresholding an image to canny detection.

-> Building the CNN model to detect each image (finally..)

	1. Trying out a basic CNN model first.
	2. A problem seems to be on the horizon : Each image that I have extracted is of different dimensions. So do I need to train the network... Maybe I found a solution, wait.
	3. Trying with EMNIST dataset first..
		○ A very promising project dealing with Capital letters and digits
		○ Eventually, might use this.
		○ Going with the handwritten digits, since no printed dataset is available (weird). 
		○ Issue with 0 and O ignored.
	4. Need to learn the OS module to import images to the notebook. Turns out to be quiet easy.
	5. Doing some initial training tests.
									Initial Results:
	
			
	6. The issue thought earlier does come up. All the extracted images are supposed to be of the same size as that of the images on which the model is trained.
	7. Planning to pad the extracted images to make them a 28x28 image. Well, did it using an inbuilt function.
	8. Almost managed to get a 100% error rate. ☺
	9. Have completed the 'stitching back the images' part, now only need to improve accuracy. 
		NOTE : Image3 is causing some issue. Have a look. Cv2.copyMakeBorder is Buggy

-> Improving the accuracy of the model.

	• Really don't think using a handwritten database is the right way ahead.
	• Will look for other models, but till then, try to improve the accuracy from 85% to at least 90% of the existing model.
	• Tried with multiple copies of the same image as database for each class, doesn't work.
	• A decent contender, but only from A-J and no digits.
	• FOUND THE MOST BEAUTIFUL CODE TO GENERATE YOUR OWN DATASET : Code
	• Well, initial modelling failed despite of incredible accuracy. Got to hold on to my gut feeling.
	• Tried a couple of more datasets to train, doesn't work.
	• Need to understand things at a grater depth.
	• Horizon looks foggy.
