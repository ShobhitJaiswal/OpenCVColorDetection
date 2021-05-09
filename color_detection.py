import numpy as np  
import cv2
import sys
import argparse
from datetime import timedelta
from datetime import datetime

def ff_to_hhmmss(ff,fps):
	return timedelta(seconds=(ff / fps))

def detectCB(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    resltn = np.shape(img)
    
    # lower_black = np.array([104, 124,  21],np.uint8)
    # upper_black = np.array([124 ,144, 101],np.uint8)

    # lower_black1 = np.array([95, 178,  0],np.uint8)
    # upper_black1 = np.array([115 ,225, 78],np.uint8)

    # lower_black1 = np.array([95, 61,  25],np.uint8)
    # upper_black1 = np.array([115 ,81, 105],np.uint8)

    # lower_blue = np.array([110,50,50],np.uint8)
    # upper_blue = np.array([130,255,255],np.uint8)

    # lower_red = np.array([0,39,64],np.uint8)
    # upper_red = np.array([10,255,255],np.uint8)

    # lower_cyan = np.array([81,39,64],np.uint8)
    # upper_cyan = np.array([100,255,255],np.uint8) 

    # lower_green = np.array([40,40,20],np.uint8)
    # upper_green = np.array([64,255,255],np.uint8)

    # lower_yellow = np.array([30,39,64],np.uint8)
    # upper_yellow = np.array([100,255,255],np.uint8)

    # lower_white = np.array([0,0,229],np.uint8)
    # upper_white = np.array([180,38,255],np.uint8)

    
    # lower_magenta = np.array([136,84,134],np.uint8)
    # upper_magenta = np.array([180,255,255],np.uint8)

    lower_pitch=np.array([22,58,166],np.uint8)
    upper_pitch=np.array([42,78,246],np.uint8)

    lower_pitch1=np.array([21,63,164],np.uint8)
    upper_pitch1=np.array([41,83,244],np.uint8)

    #finding the range of red,blue,cyan,green,yellow and white color in the image
    # black= cv2.inRange(hsv,lower_black, upper_black)
    # black1= cv2.inRange(hsv,lower_black1, upper_black1)
    pitch=cv2.inRange(hsv,lower_pitch,upper_pitch)
    pitch1=cv2.inRange(hsv,lower_pitch1,upper_pitch1)
    # red= cv2.inRange(hsv,lower_red, upper_red)
    # blue= cv2.inRange(hsv,lower_blue,upper_blue)
    # cyan= cv2.inRange(hsv,lower_cyan,upper_cyan)
    # green = cv2.inRange(hsv,lower_green,upper_green)
    # white = cv2.inRange(hsv,lower_white,upper_white)
    # yellow = cv2.inRange(hsv,lower_yellow,upper_yellow)
    # magenta = cv2.inRange(hsv,lower_magenta,upper_magenta)

    #Morphological transformation, Dilation  	
    kernal = np.ones((5 ,5), "uint8")
    # c_code = ['Red','Blue','Green','Cyan','Yellow','White','Magenta']
    c_code = ['Black','Black','Pitch']
    # black  = cv2.erode(black, kernal)
    # bbox(img, black,c_code[0],resltn)
    # res0 = cv2.bitwise_and(img, img, mask = black)
    # black1  = cv2.erode(black1, kernal)
    # bbox(img, black1,c_code[1],resltn)
    # res1 = cv2.bitwise_and(img, img, mask = black1)
    
    pitch=cv2.erode(pitch, kernal)
    bbox(img, pitch,c_code[2],resltn)
    res = cv2.bitwise_and(img, img, mask = pitch)

    pitch1=cv2.erode(pitch1, kernal)
    bbox(img, pitch1,c_code[2],resltn)
    res1 = cv2.bitwise_and(img, img, mask = pitch1)


    # red  = cv2.erode(red, kernal)
    # bbox(img, red,c_code[0],resltn)
    # res0 = cv2.bitwise_and(img, img, mask = red)

    # blue = cv2.erode(blue,kernal)
    # bbox(img, blue,c_code[1],resltn)
    # res1 = cv2.bitwise_and(img, img, mask = blue)

    # cyan = cv2.erode(cyan,kernal)
    # bbox(img, cyan,c_code[3],resltn)
    # res2 = cv2.bitwise_and(img, img, mask = cyan)    
    
    # green = cv2.erode(green,kernal)
    # bbox(img, green,c_code[2],resltn)
    # res3 = cv2.bitwise_and(img, img, mask = green)

    # white = cv2.erode(white,kernal)
    # bbox(img, white,c_code[5],resltn)
    # res4 = cv2.bitwise_and(img, img, mask = white)

    # yellow = cv2.erode(yellow,kernal)
    # bbox(img, yellow,c_code[4],resltn)
    # res5 = cv2.bitwise_and(img, img, mask = yellow)

    # magenta = cv2.erode(magenta,kernal)
    # bbox(img, magenta,c_code[6],resltn)
    # res5 = cv2.bitwise_and(img, img,mask = magenta)
    
def bbox(img, c_value, c_code, resltn):
    (_,contours,hierarchy) = cv2.findContours(c_value,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	
    #cnt = 0
    global cnt
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        width = resltn[1]
        height = resltn[0]
        # print(height*width)
        detected_area = (((height*width)*4)/100)
        detected_area1 = (((height*width)*14)/100)

        
        if area > detected_area and area < detected_area1:
            print(area,detected_area,detected_area1)
            cnt += 1
            x,y,w,h = cv2.boundingRect(contour)
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)
            cv2.putText(img,'%s'% c_code,(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),3)

def crop_frame_by_percentage(image, left=0, top=0, right=0, bottom=0):
    """
        Specify image as numpy array or image path,
        specify left, top, bottom, right percentage to be excluded in crop 

        coord = (left, top, right, bottom)
        left = 0+width*%
        right = width-width*%
        top = 0+height*%
        bottom = height-height*%
        new_img = img[top:bottom, left:right]

        new_img = img[0+ret_percent(height, 0):height-ret_percent(height, 15), 0+ret_percent(width, 10):width-ret_percent(width, 10)]
        print(0+ret_percent(height, 0), height-ret_percent(height, 15), 0+ret_percent(width, 10), width-ret_percent(width, 10))
    """
    if not isinstance(image, np.ndarray):
        image = cv2.imread(image)
    shape = image.shape
    # grayscale image
    if len(shape) == 2:
        height, width = shape
    # color image
    elif len(shape) == 3:
        height, width, channel = shape

    top = int(0+height*top*0.01)
    bottom = int(height-height*bottom*0.01)
    left = int(0+width*left*0.01)
    right = int(width-width*right*0.01)
    return image[top:bottom, left:right]              
if __name__ == '__main__': 
    # ar = argparse.ArgumentParser(description="detecting Color Bar in a video")
    # ar.add_argument("inputfilepath", type=str, help="Pass Input file directory path as argument")
    # ar.add_argument("-v", "--version", action='version', version ="Detect Color bar application v = 1.0.0 copyrights@ Planetcast Media Services Limited.")
    # arg = ar.parse_args()
    # if arg.inputfilepath:
    #     if len(sys.argv) < 2:
    #         print("Error - file name must be specified as first argument.")
    #         print("See the header of part1-threshold.py for usage details.")
    #         sys.exit()

    vid_path = r'C:\Users\Shobhit Jaiswal\Desktop\opencv-python-color-detection\Magnificent Virat Kohli Hits Brilliant Century - Windies vs India 2nd ODI 2019 - Highlights.mp4'    
    print(vid_path)
    cap = cv2.VideoCapture(vid_path)
    
    #cap.set(cv2.CAP_PROP_POS_MSEC,60)
    i =0
    frame = 0
    frame_cnt = 1
    while True:
        ret, img = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        if int(fps) != frame_cnt:
            pass
        else:
            frame += fps
            frame_cnt = 0
            #img = cv2.imread(r"E:\extract_frame\testing\output\n\image_07.jpg")
            # if not ret:
            #     break

            cnt = 0
            img_new = crop_frame_by_percentage(img,left=20, top=0, right=20, bottom=0)
            
            detectCB(img_new)
            if cnt == 1:
                time = ff_to_hhmmss(frame,fps)
                print('!! Pitch detected !!', cnt,str(time))
                cv2.imwrite(r'C:\Users\Shobhit Jaiswal\Desktop\opencv-python-color-detection\in\images_{}.jpg'.format(i),img_new)
            else:
                pass
                cv2.imwrite(r'C:\Users\Shobhit Jaiswal\Desktop\opencv-python-color-detection\out\image_{}.jpg'.format(i),img_new)
            i +=1
            cv2.imshow('frame',img_new)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        frame_cnt += 1
       