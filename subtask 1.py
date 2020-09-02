import cv2
import numpy as np

def main():
    img=cv2.imread("Task 1.png",-1)                                     
    # cv2.imshow("Original Image",img)                                    #display original image(uncomment to view)
    # cv2.waitKey(0) & 0xFF 
    grey_img=cv2.imread("Task 1.png",0)                                   #load image in grayscale
    # img=np.zeros([256,256,3],np.uint8)                                  #for testing with a circle
    # img=cv2.circle(img,(128,128),100,(255,255,255),-1)                  
    _,th=cv2.threshold(grey_img,127,255,0)                                #convert to binary so we can detect shape
    M=cv2.moments(th)                   

    x = int(M["m10"]/M["m00"])                                            #find centroid by formula of moments
    y = int(M["m01"]/M["m00"])
    print("Centroid: ")
    print("X = "+str(x)+", Y = "+str(y))

    cv2.circle(img, (x, y), 5, (255, 255, 255), -1)                     #label for visualisation. please uncomment to view
    cv2.putText(img, "centroid", (x - 25, y - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # cv2.circle(img, (x, y), 5, (0, 0, 0), -1)                           #for circle(as circle is white so label should be black) 
    # cv2.putText(img, "centroid", (x - 25, y - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # cv2.imshow("Centroid Image", img)                                    #to show image(for visualisation) please uncomemnt to see
    # cv2.waitKey(0) & 0xFF 
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()