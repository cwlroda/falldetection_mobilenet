import os
import cv2
import sys
import time
import platform
import numpy as np
from multiprocessing import Queue as pQueue
from threading import Thread
from queue import Queue, LifoQueue

class DetectionLoader:
    def __init__(self, model, data_loader, queueSize):
        self.model = model
        self.data_loader = data_loader
        self.w = self.model.getw()
        self.h = self.model.geth()
        
        self.keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']
        self.POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9], [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16], [0,15], [15,17], [2,17], [5,16]]
        self.mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], [55,56], [37,38], [45,46]]
        self.colors = [[0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255], [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255], [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]

        self.threshold = 0.2
        self.nPoints = 18
        
        self.old_neck = -1*np.ones(20, dtype=int)
        self.new_neck = -1*np.ones(20, dtype=int)
        self.subject_height = -1*np.ones(20, dtype=int)
        self.fall_ratio = 0.5
        self.fallcount = 0
        
        self.totalframecount = 0
        self.frameClone = None
        self.Q = Queue(maxsize=0)
        
    def getKeypoints(self):
        mapSmooth = cv2.GaussianBlur(self.probMap, (3, 3), 0, 0)
        mapMask = np.uint8(mapSmooth>self.threshold)
        keypoints = []
        contours = None
        try:
            #OpenCV4.x
            contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except:
            #OpenCV3.x
            _, contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            blobMask = np.zeros(mapMask.shape)
            blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
            maskedProbMap = mapSmooth * blobMask
            _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
            keypoints.append(maxLoc + (self.probMap[maxLoc[1], maxLoc[0]],))

        return keypoints
    
    def getValidPairs(self):
        valid_pairs = []
        invalid_pairs = []
        n_interp_samples = 10
        paf_score_th = 0.1
        conf_th = 0.5

        for k in range(len(self.mapIdx)):
            pafA = self.outputs[0, self.mapIdx[k][0], :, :]
            pafB = self.outputs[0, self.mapIdx[k][1], :, :]
            pafA = cv2.resize(pafA, (self.w, self.h))
            pafB = cv2.resize(pafB, (self.w, self.h))

            candA = self.detected_keypoints[self.POSE_PAIRS[k][0]]
            candB = self.detected_keypoints[self.POSE_PAIRS[k][1]]
            
            nA = len(candA)
            nB = len(candB)

            if( nA != 0 and nB != 0):
                valid_pair = np.zeros((0,3))
                for i in range(nA):
                    max_j=-1
                    maxScore = -1
                    found = 0
                    for j in range(nB):
                        d_ij = np.subtract(candB[j][:2], candA[i][:2])
                        norm = np.linalg.norm(d_ij)
                        if norm:
                            d_ij = d_ij / norm
                        else:
                            continue
                        interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                                np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                        paf_interp = []
                        for k in range(len(interp_coord)):
                            paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                            pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                        paf_scores = np.dot(paf_interp, d_ij)
                        avg_paf_score = sum(paf_scores)/len(paf_scores)

                        if (len(np.where(paf_scores > paf_score_th)[0])/n_interp_samples) > conf_th :
                            if avg_paf_score > maxScore:
                                max_j = j
                                maxScore = avg_paf_score
                                found = 1
                    if found:
                        valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

                valid_pairs.append(valid_pair)
            else:
                invalid_pairs.append(k)
                valid_pairs.append([])
        return valid_pairs, invalid_pairs
    
    def getPersonwiseKeypoints(self):
        personwiseKeypoints = -1 * np.ones((0, 19))

        for k in range(len(self.mapIdx)):
            if k not in self.invalid_pairs:
                partAs = self.valid_pairs[k][:,0]
                partBs = self.valid_pairs[k][:,1]
                indexA, indexB = np.array(self.POSE_PAIRS[k])

                for i in range(len(self.valid_pairs[k])):
                    found = 0
                    person_idx = -1
                    for j in range(len(personwiseKeypoints)):
                        if personwiseKeypoints[j][indexA] == partAs[i]:
                            person_idx = j
                            found = 1
                            break

                    if found:
                        personwiseKeypoints[person_idx][indexB] = partBs[i]
                        personwiseKeypoints[person_idx][-1] += self.keypoints_list[partBs[i].astype(int), 2] + self.valid_pairs[k][i][2]
                        
                    elif not found and k < 17:
                        row = -1 * np.ones(19)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = sum(self.keypoints_list[self.valid_pairs[k][i,:2].astype(int), 2]) + self.valid_pairs[k][i][2]
                        personwiseKeypoints = np.vstack([personwiseKeypoints, row])
                        
        return personwiseKeypoints

    def start(self):
        self.t = Thread(target=self.update(), args=(self.data_loader))
        self.t.daemon = True
        self.t.start()
        self.t.join()

        return self
    
    def update(self):
        while True:
            frame = self.data_loader.getFrame()
            
            if frame is None:
                return
            
            colw = frame.shape[1]
            colh = frame.shape[0]
            new_w = int(colw * min(self.w/colw, self.h/colh))
            new_h = int(colh * min(self.w/colw, self.h/colh))
            
            resized_image = cv2.resize(frame, (self.w, new_h), interpolation = cv2.INTER_NEAREST)
            canvas = np.full((self.h, self.w, 3), 128)
            canvas[(self.h - new_h)//2:(self.h - new_h)//2 + new_h,(self.w - new_w)//2:(self.w - new_w)//2 + new_w, :] = resized_image

            prepimg = canvas
            prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
            prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW, (1, 3, 368, 432)
            self.outputs = self.model.get_execnet().infer(inputs={self.model.get_inputblob(): prepimg})["Openpose/concat_stage7"]
            
            self.detected_keypoints = []
            self.keypoints_list = np.zeros((0, 3))
            keypoint_id = 0
            
            for part in range(self.nPoints):
                self.probMap = self.outputs[0, part, :, :]
                self.probMap = cv2.resize(self.probMap, (canvas.shape[1], canvas.shape[0])) # (432, 368)
                keypoints = self.getKeypoints()
                keypoints_with_id = []

                for i in range(len(keypoints)):
                    keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                    self.keypoints_list = np.vstack([self.keypoints_list, keypoints[i]])
                    keypoint_id += 1
            
                self.detected_keypoints.append(keypoints_with_id)
                
            self.frameClone = np.uint8(canvas.copy())
            for i in range(self.nPoints):
                for j in range(len(self.detected_keypoints[i])):
                    cv2.circle(self.frameClone, self.detected_keypoints[i][j][0:2], 5, self.colors[i], -1, cv2.LINE_AA)

            self.valid_pairs, self.invalid_pairs = self.getValidPairs()
            personwiseKeypoints = self.getPersonwiseKeypoints()
            
            # Resets points if no one is detected in the frame to prevent false positives between frames
            if range(len(personwiseKeypoints)) == 0:
                old_neck = -1*np.ones(20, dtype=int)
                new_neck = -1*np.ones(20, dtype=int)
                subject_height = -1*np.ones(20, dtype=int)

            # Fall algorithm
            # TO-DO: Optimise loops (multithreading?)
            for n in range(len(personwiseKeypoints)):
                for i in range(18):
                    index = personwiseKeypoints[n][np.array(self.POSE_PAIRS[i])]
                    
                    if -1 in index:
                        continue
                    
                    B = np.int32(self.keypoints_list[index.astype(int), 0])
                    A = np.int32(self.keypoints_list[index.astype(int), 1])
                    cv2.line(self.frameClone, (B[0], A[0]), (B[1], A[1]), self.colors[i], 3, cv2.LINE_AA)
                    
                    # Detect falling from neck points
                    if i == 0:
                        self.new_neck[n] = A[0]  
                    if i == 8:
                        self.subject_height[n] = A[0] - self.new_neck[n]
                    
                if self.totalframecount != 0 and self.totalframecount % 10 == 0:
                    if ((self.new_neck[n] - self.old_neck[n]) > self.subject_height[n]*self.fall_ratio) and self.new_neck[n] != -1 and self.old_neck[n] != -1 and self.subject_height[n] != -1:
                        self.fallcount += 1
                        print("Fall detected")
                        cv2.imwrite('output/img/'+str(self.fallcount)+'.jpg', self.frameClone)
            
                    self.old_neck[n] = self.new_neck[n]
            
            if self.fallcount != 0:
                cv2.putText(self.frameClone, "FALL COUNT: {0}".format(self.fallcount), (432-170,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
            
            self.Q.put(self.frameClone)
            self.totalframecount += 1
        
    def getFrame(self):
        # return next frame in the queue
        if self.Q.empty():
            return None
        else:
            return self.Q.get()