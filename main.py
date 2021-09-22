# import what is used
import cv2
import numpy as np
import time


cam = cv2.VideoCapture(0)

while True:
    start_time = time.time()
    # create images
    ret1, img1 = cam.read()
    time.sleep(.3)
    ret2, img2 = cam.read()
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)

    # break when a key is pressed
    if cv2.waitKey(5) >= 0:
        break

    # create orb
    orb = cv2.ORB_create(200)

    # detect keypoints
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # brute force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # i don't know what this does, but I copied and pasted and think it works
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
    cv2.imshow('matches', img3)

    p1 = [kp1[match.queryIdx].pt for match in matches[:10]]
    p2 = [kp2[match.trainIdx].pt for match in matches[:10]]

    # find the average distance between two points
    d = np.average([np.sqrt((p1[i][0] - p2[i][0])**2 + (p1[i][1] - p2[i][1])**2) for i, array in enumerate(p1)])
    theta = #TODO
    dt = time.time() - start_time
    v = d/dt
    print(v)


cam.release()
cv2.destroyAllWindows()