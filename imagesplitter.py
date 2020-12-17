import cv2 
import os 
import time
from glob import glob

currentframe = 0
outfolder = "D:\\pix2pix-Cityscapes-master-without-intermediate-checkpoints\\datasets\\facades\\test_split\\"

for path in glob("D:\\pix2pix-Cityscapes-master-without-intermediate-checkpoints\\datasets\\facades\\test\\*.jpg"):
    img = cv2.imread(path) 
    head, tail = os.path.split(path)
    outname = os.path.splitext(tail)[0]
    outputfolder = os.path.join(outfolder, outname)


    if not os.path.exists(outputfolder): 
        os.makedirs(outputfolder)
        
        # if video is still left continue creating images 
        name = str(currentframe)
        print ("Creating Frame: " + str(currentframe)) 
        crop_img1 = img[0:256, 0:256] 
        crop_img2 = img[0:256, 256:512]
        cv2.imshow("cropped", crop_img1)
        cv2.imwrite(outfolder+"\\"+name+".jpg", crop_img1)
        cv2.imwrite(outfolder+"\\"+name+"_label.jpg", crop_img2)
        cv2.imshow("frame", crop_img2)
        cv2.waitKey(1)

        currentframe += 1

cv2.destroyAllWindows()