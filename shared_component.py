#import libraries
import openpyxl
import pandas as pd
from openpyxl import load_workbook
from openpyxl import Workbook
import glob
import re
import math
import cv2

#get folder of images
folder = 'Results'
images = glob.glob(folder + '/*.jpg')


#read in the data files
od = pd.read_excel("DCDCtgpz35/result/od.xlsx")
os = pd.read_excel("DCDCtgpz35/result/os_cleaned.xlsx")

#set headers for new csv
headers = ["image", "score"]

#cretae empty dataframe
df = pd.DataFrame(columns=headers)

#loop through all images
for image in images:
    
    #regex pattern to match filnames
    pattern = r"RET(\d{3})([A-Z]{2})"
    matches = re.search(pattern, image)

    if matches:

        #get id from regex
        number = matches.group(1)
        id = '#' + str(number)

        #get dataset from regex
        letters = matches.group(2) 
        file = letters.lower()

        #read in image
        img = cv2.imread(image)

        #get image dimensions for cropping
        height, width = img.shape[:2]

        #get centre for crop
        center_x, center_y = int(width / 2), int(height / 2)

        if file == 'od':

            #get axial length for id
            row = od.loc[od['Unnamed: 0'] == id]
            axial_length = row['Axial_Length'].values[0]

            #calulate ratio and crop dimension
            ratio = axial_length/26
            dimension = 128 * ratio
            int_dimension = int(dimension)

            #get cropped dimensions
            crop_width, crop_height = int_dimension, int_dimension
            crop_x, crop_y = center_x - int(crop_width / 2), center_y - int(crop_height / 2)

            #crop image
            cropped_img = img[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]

            #greyscale for otsu threshhold
            gray_cropped = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

            #otsu threshold
            ret, thresh = cv2.threshold(gray_cropped, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            #divide white pixels by total pixels
            white_num = cv2.countNonZero(thresh)
            total = thresh.shape[0] * thresh.shape[1]
            white_ratio = white_num / total

            #add image and score to dataframe
            name = image[8:]
            new_row = {'image': name, 'score': white_ratio}
            df.loc[len(df)] = new_row

        else:

            #get axial length for id
            row = os.loc[os['Unnamed: 0'] == id]
            axial_length = row['Axial_Length'].values[0]

            #calulate ratio and crop dimension
            ratio = axial_length/26
            dimension = 128 * ratio
            int_dimension = int(dimension)


            #get cropped dimensions
            crop_width, crop_height = int_dimension, int_dimension
            crop_x, crop_y = center_x - int(crop_width / 2), center_y - int(crop_height / 2)

            #crop image
            cropped_img = img[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
        
            #greyscale for otsu threshhold
            gray_cropped = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

            #otsu threshhold
            ret, thresh = cv2.threshold(gray_cropped, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            #divide white pixels by total pixels
            white_pixels = cv2.countNonZero(thresh)
            total_pixels = thresh.shape[0] * thresh.shape[1]
            white_ratio = white_pixels / total_pixels

            #add image and score to dataframe
            name = image[8:]
            new_row = {'image': name, 'score': white_ratio}
            df.loc[len(df)] = new_row

#save to csv
df.to_csv('score_results.csv', index = False)