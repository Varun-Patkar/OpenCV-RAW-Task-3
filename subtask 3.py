import cv2
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from math import log10, sqrt 
import numpy as np
from matplotlib import pyplot as plt

def median_filter(img):                                                            #median filter 
    return cv2.medianBlur(img, 3)

def PSNR(original, compressed):                                                    #calculate RMSE and PSNR
    mse=np.mean((original-compressed)**2) 
    if(mse==0):  
        return 100
    max_pixel=255.0
    psnr=20*log10(max_pixel/sqrt(mse)) 
    return (sqrt(mse),psnr) 

def main():
    img=cv2.imread("Task 3.jpg")                                                   #read and show image to user
    cv2.imshow("Original Image",img)
    cv2.waitKey() & 0xFF

    noisy=img[861:924,204:308,:]                                                   #cut out a small uniform part and draw its histogram. Shows Gaussian distribution
    plt.title("Histogram of Clean grey area to estimate type of noise 1")
    plt.hist(noisy.ravel(),256,[0,256])
    plt.show()

    noisy=img[704:807,1108:1182,:]                                                 #for sanity check, cut out another uniform part and check. Shows Gaussian PDF 
    plt.title("Histogram of Clean grey area to estimate type of noise 2")
    plt.hist(noisy.ravel(),256,[0,256])
    plt.show()

    print("\nMETRICS USED:\nPSNR:HIGHER IS BETTER \nRMSE:LOWER IS BETTER\n")       #higher PSNR(calculates similarity with original image) and lower RMSE means better denoised image

    sp_img=median_filter(img)                                                      #Apply Median Filter and print results and show image
    rmse_sp,psnr_sp=PSNR(sp_img,img)
    print("Median Filter PSNR : "+str(psnr_sp))                                    #Median Filter PSNR : 42.74279602338066
    print("Median Filter RMSE : "+str(rmse_sp)+"\n")                               #Median Filter RMSE : 1.8595179683597514
    cv2.imshow("Median Filter Test", sp_img)
    cv2.waitKey() & 0xFF

    bilateral_img=cv2.bilateralFilter(img,3,100,100)                               #Apply Bilateral Filter and print results and show image(BEST PSNR AND RMSE RATING)
    rmse_bilateral,psnr_bilateral=PSNR(bilateral_img,img)
    print("Bilateral Filter PSNR : "+str(psnr_bilateral))                          #Bilateral Filter PSNR : 45.270338974238484
    print("Bilateral Filter RMSE : "+str(rmse_bilateral)+"\n")                     #Bilateral Filter RMSE : 1.3900269859615635
    cv2.imshow("Bilateral Filter Test", bilateral_img)
    cv2.waitKey() & 0xFF

    gaussian_img=cv2.GaussianBlur(img,(3,3),0)                                     #Apply Gaussian Filter and print results and show image
    rmse_gaussian,psnr_gaussian=PSNR(gaussian_img,img)
    print("Gaussian Filter PSNR : "+str(psnr_gaussian))                            #Gaussian Filter PSNR : 43.963393881862594
    print("Gaussian Filter RMSE : "+str(rmse_gaussian)+"\n")                       #Gaussian Filter RMSE : 1.6157363154753417
    cv2.imshow("Gaussian Filter Test", gaussian_img)
    cv2.waitKey() & 0xFF

    nonlocal_img=cv2.fastNlMeansDenoisingColored(img,h=1)                          #Apply Non Local Filter and print results and show image
    rmse_nonlocal,psnr_nonlocal=PSNR(nonlocal_img,img)
    print("Non-Local Means Filter PSNR : "+str(psnr_nonlocal))                     #Non-Local Means Filter PSNR : 44.68882223740083
    print("Non-Local Means Filter RMSE : "+str(rmse_nonlocal)+"\n")                #Non-Local Means Filter RMSE : 1.4862746433745664
    cv2.imshow("Non-Local Means Filter Test", nonlocal_img)
    cv2.waitKey() & 0xFF

    arithmean = cv2.boxFilter(img, -1, (3, 3))                                     #Apply Arithmetic Mean Filter and print results and show image
    rmse_arithmean,psnr_arithmean=PSNR(arithmean,img)
    print("Arithmetical Mean Filter PSNR : "+str(psnr_arithmean))                  #Arithmetical Mean Filter PSNR : 41.819607175768745
    print("Arithmetical Mean Filter RMSE : "+str(rmse_arithmean)+"\n")             #Arithmetical Mean Filter RMSE : 2.068044223872488
    cv2.imshow("Arithmetical Mean Filter Test", arithmean)
    cv2.waitKey() & 0xFF

    log=np.float32(np.log(img))                                                    #Apply Geometric Mean Filter and print results and show image
    geomean2 = np.uint8(np.exp(cv2.boxFilter(log, -1, (3, 3))))
    rmse_geomean,psnr_geomean=PSNR(geomean2,img)
    print("Geometrical Mean Filter PSNR : "+str(psnr_geomean))                     #Geometrical Mean Filter PSNR : 41.47952729296938
    print("Geometrical Mean Filter RMSE : "+str(rmse_geomean)+"\n")                #Geometrical Mean Filter RMSE : 2.150620671069943
    cv2.imshow("Geometrical Mean Filter Test", geomean2)
    cv2.waitKey() & 0xFF

    harmmean = 1/cv2.boxFilter(1/img, -1, (3, 3))                                  #Apply Harmonic Mean Filter and print results and show image
    harmmean=np.uint8(harmmean)
    rmse_harmmean,psnr_harmmean=PSNR(harmmean,img)
    print("Harmonic Mean Filter PSNR : "+str(psnr_harmmean))                       #Harmonic Mean Filter PSNR : 41.336710973538416
    print("Harmonic Mean Filter RMSE : "+str(rmse_harmmean)+"\n")                  #Harmonic Mean Filter RMSE : 2.1862742101289583
    cv2.imshow("Harmonic Mean Filter Test", harmmean)
    cv2.waitKey() & 0xFF

    final=np.concatenate((img[743:,996:,:],sp_img[743:,996:,:],gaussian_img[743:,996:,:],nonlocal_img[743:,996:,:],arithmean[743:,996:,:],geomean2[743:,996:,:],harmmean[743:,996:,:],bilateral_img[743:,996:,:]),axis=1)
    cv2.imshow("Comparison of All", final)                                         #stitch a small corner of all images together for visual check for noise
    cv2.waitKey() & 0xFF

    #On checking visually it was found that the best denoising techique was Non Local filter. But on analysing its output image, I saw that it has a significant loss in detail
    #This is not what we want. By looking at all the pictures indivisually and analysing the PSNR and RMSE scores it is clear that the best filter is Bilateral Filter
    #This is because it manages to remove a lot of noise but still has some noise but there is no significant loss in detail. So by keeping the balance Bilateral filter works the best
    
    #In terms of pure denoising: Non local filter is best as the noise is very less.
    #In terms of denoising as well as retaining detail: Bilateral Filter works the best(it removes a lot of noise while retaining all the detail)

    #We dont want loss in detail as if we take the low detail picture after this preprocessing step, the results of edge detection and all other algorithms will be fuzzy and inaccurate. 
    #Hence balance in detail and denoising should be maintained

    cv2.destroyAllWindows()


if __name__=="__main__":
    main()