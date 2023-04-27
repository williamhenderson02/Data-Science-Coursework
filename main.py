import cv2
import glob
import sys
import os
import numpy as np
import math

#get folder argument from command line
folder = sys.argv[1]

#create Results folder to store images
if not os.path.exists('Results'):
    os.makedirs('Results')

#get pathname for images
images = glob.glob(folder + '/*.jpg')

#function for median filter
def median_filter(img):

    #remove salt and pepper noise
    median = cv2.medianBlur(img,3)
    
    return median

#function for de noising coloured images
def denoising_coloured(img):

  #Non-local-means filter to remove gaussian noise
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 1, 11, 3, 13)

    return denoised

def gamma_correction(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Compute average brightness of image
    v_mean = np.mean(hsv[:,:,2])

    # Set gamma value based on average brightness
    if v_mean < 127:
        
        gamma = 0.5
    else:
        gamma = 1.5

    # Apply gamma correction to V channel
    v_corrected = np.power(hsv[:,:,2] / 255.0, gamma) * 255.0

    # Clip values to 0-255 range
    v_corrected = np.clip(v_corrected, 0, 255).astype(np.uint8)

    # Merge corrected V channel back into HSV image
    hsv[:,:,2] = v_corrected

    # Convert image back to RGB color space
    gamma_adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return gamma_adjusted

def contrast_stretching(img):

    # Convert image to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split LAB channels
    L, A, B = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L = clahe.apply(L)

    # Merge LAB channels
    lab = cv2.merge((L, A, B))

    # Convert LAB image back to BGR color space
    stretched = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return stretched

#function to dewarp by perspective
def dewarp_perspective(img):

    #define source points for perspective transformtion
    src_points = np.float32([(0, 0), (img.shape[1], 0), (img.shape[1], img.shape[0]), (0, img.shape[0])])

    #define destination points for perspective transformation
    dst_points = np.float32([(0, 0), (img.shape[1], 0), (img.shape[1], img.shape[0] * 0.8), (0, img.shape[0] * 0.8)])

    #create matrix for transformations
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    #apply transformation
    dewarped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

    return dewarped

def inpaint(img):

    #create gray image
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    #use exemplar based inpainting
    inpainted = cv2.inpaint(dewarped, gray, 10, cv2.INPAINT_TELEA)

    return inpainted

#loop through each image
for image in images:

    #read the image
    img = cv2.imread(image)

    mask = cv2.imread('custom_mask.jpeg')    

    #calling each function for enhancments
    median = median_filter(img)
    denoised_coloured = denoising_coloured(median)
    stretched = contrast_stretching(denoised_coloured)
    gamma_adjusted = gamma_correction(stretched)
    dewarped = dewarp_perspective(gamma_adjusted)
    inpainted = inpaint(dewarped)
 
    #cv2.imshow("mask", mask)
    #cv2.imshow('start', img)
    #cv2.imshow('median',median)
    #cv2.imshow('Denoised', denoised_coloured)
    #cv2.imshow('Gamma', gamma_adjusted)
    #cv2.imshow('Stretch', stretched)
    #cv2.imshow('Dewarped', dewarped)
    #cv2.imshow('Processed Image', inpainted)

    #save image to Results folder
    processed_image_path = os.path.join('Results', os.path.basename(image))
    cv2.imwrite(processed_image_path, inpainted)