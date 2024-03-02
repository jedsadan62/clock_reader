import cv2
import numpy as np
import os
import imutils

def nothing(x):
    pass

def splitZ(position,frame):
    list = []
    for i in position:
        crop_image= frame[i[1]:i[1]+i[3],i[0]:i[0]+i[2]]
        list.append(crop_image)
    return list

def wordAnalysis(word):
    # data folser path
    data_folder_path = '/Users/mac-ice/Al/alphabet_card'
    labelList = []
    name = [None,None]
    for u in word:
        word1 = cv2.cvtColor(u, cv2.COLOR_BGR2GRAY)
        _,word1= cv2.threshold(word1,127,255,cv2.THRESH_BINARY_INV)
        (iH,iW) = word1.shape
        res_threshold_list = []
        for dataset_file in os.listdir(data_folder_path):
            template = cv2.imread("{0}/{1}".format(data_folder_path,dataset_file))
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            (thresh,template) = cv2.threshold(template,127,255,cv2.THRESH_BINARY_INV)
            (tH,tW) = template.shape[:2] # each card minimum size is H:44pix x W:31pix
            for scale in np.linspace(0.1,5.0,100):
                h = int(tH*scale)
                w = int(tW*scale)
                resized_tpl = cv2.resize(template,(w,h),interpolation = cv2.INTER_AREA)
                if (h > iH) or (w >iW) or (h<5) or (w<5):
                    continue
                res = cv2.matchTemplate(word1,resized_tpl,cv2.TM_CCOEFF_NORMED)
                # specify a threshold
                threshold = 0.89
                # store the coordinates of matched area in a numpy
                (ys , xs) = np.where(res >= threshold)
                for x,y in zip(xs , ys):
                #print("{0}".format(dataset_file),res[y][x])
                    dataset_file = dataset_file.replace("_", " ")
                    dataset_file = dataset_file.replace(".jpg", "")
                    res_threshold_list.append((res[y][x],dataset_file))
        if len(res_threshold_list)>0:
            name = max(res_threshold_list)
        if len(name)>0:
            labelList.append(name[1])
    return labelList

def find_word(frame):
    g_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    binary1 = cv2.Canny(g_image,50,100)
    position = []          
    contours,_= cv2.findContours(binary1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # print('Area = ',area)#area ที่หาได้
        # จะหาวัตถุที่ area > 1000
        if area > 1500:
            # print('Area = ',area)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            x,y,w,h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,0,255),2)
            position.append([x,y,w,h])
    position.sort(reverse=False) 
    return position




vid = cv2.VideoCapture(0)
camera_address = "http://10.59.236.243:8080/video" #กรณีใช้มือถือแทนเว็บแคม
vid.open(camera_address)
cv2.namedWindow('image')
cv2.createTrackbar('T', 'image', 0, 255, nothing) #สร้างTrackbarไว้ปรับ threshold
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()#อ่านภาพจากวิดีโอ
    # frame = cv2.imread('al.jpg')
    frame = cv2.resize(src=frame,dsize = None,fx=0.5,fy=0.5,interpolation= cv2.INTER_LINEAR)
    # frame = cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
    # kernel = np.array([[0, -1, 0],
    #                [-1, 5,-1],
    #                [0, -1, 0]])
    # frame = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)
    imgContour =  frame.copy()#อ่านภาพจากวิดีโอ Copy ภาพ
    # (iH,iW) = frame.shape[:2]#ค่าความสูง ความกว้าง
   
    word = []
    t = cv2.getTrackbarPos('T', 'image') #ไว้ปรับ threshold
    binary1 = cv2.Canny(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),t,t/2)
    position = find_word(frame)
    # _,binary1= cv2.threshold(g_image,127,255,cv2.THRESH_BINARY) #เปลี่ยภาพเป็น Binary
    # (thresh,binary1) = cv2.threshold(g_image,127,255,cv2.THRESH_BINARY_INV)
    word = splitZ(position,frame)
    if len(word) > 0:
        # labelList =[]
        labelList = wordAnalysis(word)
        print('labelList = ',labelList)

    
    # Display the resulting frame
    cv2.imshow('frame_normal', imgContour)
    cv2.imshow('frame_binary', binary1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()