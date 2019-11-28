import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


fig = plt.figure()
cap = cv.VideoCapture(0)
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 1000,
                       qualityLevel = 0.4,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(1000,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
#old_frame = cv.rotate(old_frame,cv.ROTATE_180)

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
while(1):
    try:
        ret,frame = cap.read()
        #frame = cv.rotate(frame,cv.ROTATE_180)

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        # draw the tracks
        old_points = []
        new_points = []

        prev_dir = np.array([0,0])

        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            new_points.append([a,b])


            c,d = old.ravel()
            old_points.append([c,d])

            # print(b)
            # mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            # frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)

        new_points = np.array(new_points)
        old_points =np.array(old_points)

        direction_vectors = new_points - old_points
        median_directions = np.median(direction_vectors,axis=0)

        old_median = np.median(old_points,0)
        new_median = np.median(new_points,0)
        median_directions = median_directions/np.linalg.norm(median_directions)
        median_directions[0] *= -1
        alpha = 0.7
        median_directions = alpha * median_directions + (1-alpha) * prev_dir
        # plt.quiver([0,0],median_directions[0],median_directions[1])

        dir_mask = np.zeros((50,50))
        sum1=0
        sum2=0
        for i in range(0,30):
            sum1+=median_directions[0]
            sum2+=median_directions[1]

        median0 = sum1/30
        median1 = sum2/30

        #left is -0.4 and right is 0.4 else straight
        if(median0>=0.64):
            cv.line(dir_mask,(25,50),(100,150),(255,255,255),3)
            print("right")
        elif(median0<=-0.64) :
            cv.line(dir_mask,(25,50),(-100,150),(255,255,255),3)
            print("left")
        else:
            cv.line(dir_mask,(25,50),(25,70),(255,255,255),3)
            print("straight")

        cv.line(mask,(old_median[0],old_median[1]),(new_median[0],new_median[1]),(255,0,0),3)




        img = cv.add(frame,mask)
        cv.imshow('frame',frame)
        cv.imshow('dir',dir_mask)
        # cv.imshow('field',mask)

        k = cv.waitKey(30) & 0xff


        # print(p1.shape)
        # plt.pause(0.0005)
        #print(median_directions[0])

        prev_dir = median_directions



        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
    except:
        print('Some Error')
