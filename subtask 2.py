import cv2
import numpy as np

def main():
    img=cv2.imread("Task 2.jpg",-1)                                        #read and show original image
    cv2.imshow("Original image",img)
    cv2.waitKey(0) & 0xFF      
    grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                              #convert to greyscale and show
    cv2.imshow("GreyScale Image",grey)  
    cv2.waitKey(0) & 0xFF
    _,binary=cv2.threshold(grey,127,255,0)                                 #convert to binary and show
    cv2.imshow("Binary Image",binary) 
    cv2.waitKey(0) & 0xFF 
    cv2.destroyAllWindows()
if __name__=="__main__":
    main()