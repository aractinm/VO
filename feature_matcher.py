import cv2

import numpy as np
import argparse
import imutils
import glob
from math import sqrt
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
    
    surf = cv2.xfeatures2d.SURF_create() 
    kp1, des1 = surf.detectAndCompute(img1,None)
    kp2, des2 = surf.detectAndCompute(img2,None)
    print type(kp1), type(des1)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params  = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=10)   # or pass empty dictionary
    flann    = cv2.FlannBasedMatcher(index_params,search_params)
    matches  = flann.knnMatch(des1,des2,k=2)
    matchArr = [0]*len(matches)
    
    coords1  = []
    coords2  = []            #list that will hold the coordinates
    
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.6*n.distance:
            matchArr[i] = 1
            coords2.append(kp2[m.trainIdx].pt)     
            coords1.append(kp1[n.queryIdx].pt)
    
    npcoords1 = np.float32(coords1)
    npcoords2 = np.float32(coords2)

    F = findFundamental(npcoords1,npcoords2)    
    E = findEssential(F)
    #print E1

    R1 , t1 = decompose_e_byfunc(E,npcoords1,npcoords2)
    print 'first E\n', E
    print 'r from recover pose\n' , R1
    print 't from recover pose\n' , t1
    print np.linalg.det(R1)

#SECOND ESSENTIAL

    E2 , m = cv2.findEssentialMat(npcoords1,npcoords2,camera_matrix)
    R2 , t2 = decompose_e_byfunc(E2,npcoords1,npcoords2)
    print 'inliners', sum(m) , len(m) , len(coords1)
    print 'second essential\n' , E2
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
print 'The average time of running the program',N,'times was', sum(timelist)/N