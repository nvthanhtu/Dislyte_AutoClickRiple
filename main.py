import cv2 as cv
import time
import random

import numpy as np
import utility
import sys
from PIL import ImageGrab

area1 = (0,400,500,900)
area2 = (1902,400,500,900)
max_tried_img = 5
threshold=0.75

def main():
    # BGR 
    dhalia_sample = cv.imread('Image/Dhalia.png', cv.IMREAD_COLOR)
    # cv.imshow("result",dhalia_sample)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    
    # BGR 
    yesuhua_sample = cv.imread('Image/YeSuhua.PNG', cv.IMREAD_COLOR)
    # cv.imshow("result",yesuhua_sample)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    
    # BGR 
    meredith_sample = cv.imread('Image/Meredith.PNG', cv.IMREAD_COLOR)
    # cv.imshow("result",yesuhua_sample)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    
    # BGR 
    notnow_sample = cv.imread('Image/NotNow.PNG', cv.IMREAD_COLOR)
    # cv.imshow("result",notnow_sample)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    
    # BGR 
    confirm_sample = cv.imread('Image/confirm.PNG', cv.IMREAD_COLOR)
    # cv.imshow("result",notnow_sample)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    
    # printscreen =  np.array(ImageGrab.grab(bbox=area1))
    # printscreen = cv.cvtColor(printscreen, cv.COLOR_RGB2BGR)
    # cv.imshow("result",printscreen)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    
    while True:
        try:
            
            # RGB 
            printscreen_PIL =  np.array(ImageGrab.grab(bbox=area1))
            printscreen = cv.cvtColor(printscreen_PIL, cv.COLOR_RGB2BGR) 
            
            # print("Looking for Dhalia...")
            # check for dhalia
            sample = meredith_sample
            result = cv.matchTemplate(printscreen, meredith_sample, cv.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv.minMaxLoc(result)
            print("Matching Meredith {:0.2f}".format(max_val), end='\r')
            
            if max_val < threshold:
                # print("Looking for Ye Suhua...")
                #check for ye suhua
                sample = yesuhua_sample
                result = cv.matchTemplate(printscreen, yesuhua_sample, cv.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv.minMaxLoc(result)
                #print("Matching Ye Suhua {:0.2f}".format(max_val), end='\r')
              
            if max_val < threshold:
                #check if notnow is shown
                sample = notnow_sample
                result = cv.matchTemplate(printscreen, notnow_sample, cv.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv.minMaxLoc(result)
                
            if max_val < threshold:
                #check if confirm is shown
                sample = confirm_sample
                result = cv.matchTemplate(printscreen, confirm_sample, cv.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv.minMaxLoc(result)
            
            if max_val >= threshold:
                #print("Matching : {:0.2f}", format(max_val))
                template_w = sample.shape[1]
                template_h = sample.shape[0]

                #take into account offset when screaning
                click_position_x = int(max_loc[0] + template_w/2 + random.randint(0, 9)+ area1[0])
                click_position_y = int(max_loc[1] + template_h/2 + random.randint(0, 9) + area1[1])
                
                utility.click(click_position_x, click_position_y)
                
        except cv.error as e:
            print(e)
        
if __name__ == "__main__":
    main()
