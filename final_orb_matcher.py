import cv2
import numpy as np
# import argparse
# import imutils
# import glob
# from math import sqrt
import time
Wt = np.array([[0,1,0],[-1,0,0],[0,0,1]])
W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
camera_matrix = np.array([  [322.109684, 0.000000, 344.923419],
                            [0.000000, 304.886662, 231.220003],
                            [0.000000, 0.000000, 1.000000]        ])

def findFundamental(coords1,coords2):
    F, mask = cv2.findFundamentalMat(coords1,coords2,cv2.FM_LMEDS)
    return F

def findEssential(F):
    E = np.dot(camera_matrix.T,np.dot(F,camera_matrix))
    return E

def decompose_e(E):
    u, s, vt = np.linalg.svd(E)
    s[-1] = 0
    R1 = np.dot(u,np.dot(Wt,vt))
    R2 = np.dot(u,np.dot(W,vt))
    r1 = np.linalg.det(R1)
    r2 = np.linalg.det(R2)    

    t = np.dot(u , np.dot(W,np.dot(np.diag(s),u.T)))
    #print 'col 2 of U' , u[:,2]
    # print 'r1' , R1 , r1
    # print 'r2' , R2 , r2
    #t =u[:,2]
    return R1, R2 , t , u[:,2],r1         #!!!!!!!!!!!!!!!!!!!return t and r
def decompose_e_byfunc(E,coords1,coords2):
    points, R, t, mask = cv2.recoverPose(E, coords1, coords2)#, camera_matrix, R, t, focal )
    # print 'points\n' , points
    # print 'mask', mask
    return R , t
def normalize(coords,height,width):
    new_coords = []
    for i in range(len(coords)):
        new_coords.append([coords[i][0]/width,coords[i][1]/height])
    return new_coords

def feature_match ():
    img1 = cv2.imread('webcam_image1.jpeg',0)          # queryImage
    img2 = cv2.imread('webcam_image2.jpeg',0)          # trainImage
    # Initiate SIFT detector
    orb = cv2.ORB_create() 
    # find the keypoints with ORB
    kp1 = orb.detect(img1,None)
    kp2 = orb.detect(img2,None)
    #print cv2.drawMatchesKnn.__doc__
    # compute the descriptors with ORB
    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)
    # print len(matches), len(des1) , len(des2)
    #print len(matches)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    
    #matches = matches[:8]
    
    coords2 = []
    coords1 = []
    for i,n in enumerate(matches):
        #print n
        coords2.append(kp2[n.trainIdx].pt)     #matched keypoints of second image
        coords1.append(kp1[n.queryIdx].pt)     #matched keypoints of first image
    # h , w  =  img2.shape
    # coords1 = normalize(coords1,h,w)
    # coords2 = normalize(coords2,h,w)
    # print coords1 , coords2
    npcoords1 = np.float32(coords1)
    npcoords2 = np.float32(coords2)
    #print npcoords2
    F = findFundamental(npcoords1,npcoords2)
    
    E = findEssential(F)
    #print E1


    R , t = decompose_e_byfunc(E,npcoords1,npcoords2)
    print 'r from recover pose\n' , R
    print 't from recover pose\n' , t
    print np.linalg.det(R)

    E2 , m = cv2.findEssentialMat(npcoords1,npcoords2,camera_matrix)
    R2 , t2 = decompose_e_byfunc(E2,npcoords1,npcoords2)
    print 'inliners' , sum(m) , len(m) , len(coords1)
    print 'second essential' , E2
    print 'r2\n' , R2
    print 't2\n' , t2
#TEST 20 times
timelist = []
N = 1
for i in range(0,N,1):
    start = time.time()
    feature_match()
    end   = time.time()
    timelist.append(end-start)
print 'The average time of running the program' ,N,'times was', sum(timelist)/N