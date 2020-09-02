import cv2
import numpy as np
from matplotlib import pyplot as plt

def preprocess(img):                                              #preprocess function found at https://stackoverflow.com/questions/8753833/exact-skin-color-hsv-range#:~:text=The%20skin%20in%20channel%20H,for%20small%20values%20of%20V.
    kernel=np.ones((5,5),np.uint8)                                #the function takes the image, applies a Gaussian Blur of 3x3 for better detection, then uses a mask specially designed for white human skin 
    blur=cv2.GaussianBlur(img,(3,3), 0)                           #this might not work for African, African American, Red Indian and people of different complexion
    hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)                       #it then blurs the mask further with a median blur of 5x5 
    lower_color=np.array([108,23,82])                             #then it dilates it so that gaos get filled in
    upper_color=np.array([179,255,255])
    mask=cv2.inRange(hsv,lower_color,upper_color)
    blur=cv2.medianBlur(mask,5)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    hsv_d=cv2.dilate(blur,kernel)
    hsv_d[:328,:252]=0                                            #this is manually added to remove noise and is specific to this image
    hsv_d[:,1081:]=0                                              #this wont work for any other image
    opening=cv2.morphologyEx(hsv_d,cv2.MORPH_OPEN,kernel)         #then i use opening function to remove false positives from the image
    closing=cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel)      #then i use closing on the opening mask to remove false negatives
    return closing                                                #the function the returns the final mask

def fillHole(im_in):                                              #this function fills up holes in the mask by using the floodfill function 
	im_floodfill=im_in.copy()
	h,w=im_in.shape[:2]
	mask=np.zeros((h+2,w+2),np.uint8)
	cv2.floodFill(im_floodfill,mask,(0,0),255);
	im_floodfill_inv=cv2.bitwise_not(im_floodfill)
	im_out=im_in|im_floodfill_inv
	return im_out

def kmeans(img):                                                  #this was my try at using a k means clustering algorithm to seperate out the hand. It failed terribly as the background also was coming into the same cluster
    k=2                                                           #i used k=2-10 but none clustered the hand as a whole. Hence i commented out its code in the main function
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixel_values=img.reshape((-1, 3))
    pixel_values=np.float32(pixel_values)
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _,labels,(centers)=cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers=np.uint8(centers)
    labels=labels.flatten()
    segmented_image=centers[labels.flatten()]
    segmented_image=segmented_image.reshape(img.shape)
    masked_image=np.copy(img)
    masked_image=masked_image.reshape((-1, 3))
    cluster=1
    masked_image[labels==cluster]=[0, 0, 0]
    masked_image=masked_image.reshape(img.shape)
    return masked_image

def histogram_backprop(roi,img):                                  #i tried using histogram backpropagation but as this was based on colour segmentation, the hand was not being fully seperated
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)                       #i also commented out this code in main function. 
    hsvt=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    M=cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
    I=cv2.calcHist([hsvt],[0, 1], None, [180, 256], [0, 180, 0, 256] )
    h,s,v=cv2.split(hsvt)
    R=M/I
    B=R[h.ravel(),s.ravel()]
    B=np.minimum(B,1)
    B=B.reshape(hsvt.shape[:2])
    disc=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cv2.filter2D(B,-1,disc,B)
    B=np.uint8(B)
    cv2.normalize(B,B,0,255,cv2.NORM_MINMAX)
    _,th=cv2.threshold(B,50,255,0)
    output=cv2.bitwise_and(img,img,mask=th)
    return output

def main():
    img=cv2.imread('Task 4.jpg')                                          #read image
    
    mask=preprocess(img)                                                  #Turns out that the best result was by a code thati found on stack overflow. So i used this and commenterd out all the other code in main function
    mask=fillHole(mask)                                                   #use the fillhole function to remove holes in the mask
    output=cv2.bitwise_and(img,img,mask=mask)

    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)                          #i tried basic mask at the start with only hsv colors. An obvious failure
    # mask = cv2.inRange(hsv,(0, 10, 60), (20, 150, 255) )
    
    # output=kmeans(img)                                                  #k means code commented out(go up to know the reason)
    # output=cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
    # _,th=cv2.threshold(output,0,255,cv2.THRESH_BINARY)
    # output=cv2.bitwise_and(img,img,mask=th)

    # roi=output[437:777,238:589,:]                                       #histogram backprop comemnted out
    # output=histogram_backprop(roi,img)

    # output=cv2.circle(output,(941,394),287,(0,0,0),-1)                  #as the lens was parallel to the camera lens at time of capturing it was a perfect circle so i just calculated the centre and radius
                                                                          #and put a black circle in. It worked surprisingly well
    cv2.imshow("Output", output)                                          #show the output
    cv2.waitKey() & 0xFF
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()