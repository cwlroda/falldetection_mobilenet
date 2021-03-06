import sys
import os
import numpy as np
import cv2
from os import system
import io
import time
from os.path import isfile
from os.path import join
import re
import argparse
import platform
import warnings
import pafy

try:
    from armv7l.openvino.inference_engine import IECore, IEPlugin
except:
    from openvino.inference_engine import IECore, IEPlugin
    
warnings.filterwarnings("ignore")

def getKeypoints():

    mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)
    mapMask = np.uint8(mapSmooth>threshold)
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
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints


def getValidPairs():
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.5

    for k in range(len(mapIdx)):
        pafA = outputs[0, mapIdx[k][0], :, :]
        pafB = outputs[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (w, h))
        pafB = cv2.resize(pafB, (w, h))

        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        
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


def getPersonwiseKeypoints():
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]
                    
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
                    
    return personwiseKeypoints


if __name__ == "__main__":
    camera_width = 1080
    camera_height = 720

    keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']
    POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9], [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16], [0,15], [15,17], [2,17], [5,16]]
    mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], [55,56], [37,38], [45,46]]
    colors = [[0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255], [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255], [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", help="Specify the target device to infer on; CPU, GPU, MYRIAD is acceptable. (Default=CPU)", default="CPU", type=str)
    parser.add_argument("-b", "--boost", help="Setting it to True will make it run faster instead of sacrificing accuracy. (Default=False)", default=False, type=bool)
    # parser.add_argument("-m", "--mode", help="Mode 0: Pose tracking. Mode 1: Fall detection. (Default=0)", default=0, type=int)
    parser.add_argument("-v", "--video", help="Specify video file, if any, to perform pose estimation (Default=Webcam)", default='webcam', type=str)
    parser.add_argument("-o", "--output_dir", help="Specify output directory. (Default=\{CURR_DIR\}/output/)", default="output", type=str)
    args = parser.parse_args()

    if "webcam" == args.video:
        try:
            cap = cv2.VideoCapture(-1)
        except:
            print("Camera not found. Is your camera set up properly?")
            sys.exit(0)
            
    elif "livefeed" == args.video:
        try:
            url = input()
            video = pafy.new(url)
            best = video.getbest(preftype="mp4")
            
            cap = cv2.VideoCapture()
            cap.open(best.url)
        except:
            print("URL not found. Is the link provided correct?")
            sys.exit(0)
            
    else:
        try:
            cap = cv2.VideoCapture(args.video)
        except:
            print("Video file not found. Did you specify the correct path?")
            sys.exit(0)
    
    # Save output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_dir+'/output.mp4', fourcc, 60, (432, 368))
    
    fps = ""
    detectfps = ""
    framecount = 0
    totalframecount = 0
    time1 = 0
    
    plugin = IEPlugin(device=args.device)

    # TO-DO: Try another model to compare accuracy
    if "CPU" == args.device:
        if platform.processor() == "x86_64":
            plugin.add_cpu_extension("lib/libcpu_extension.so")
        if args.boost == False:
            model_xml = "models/train/test/openvino/mobilenet_v2_1.4_224/FP32/frozen-model.xml"
        else:
            model_xml = "models/train/test/openvino/mobilenet_v2_0.5_224/FP32/frozen-model.xml"

    elif "GPU" == args.device or "MYRIAD" == args.device:
        if args.boost == False:
            model_xml = "models/train/test/openvino/mobilenet_v2_1.4_224/FP16/frozen-model.xml"
        else:
            model_xml = "models/train/test/openvino/mobilenet_v2_0.5_224/FP16/frozen-model.xml"

    else:
        print("Specify the target device to infer on; CPU, GPU, MYRIAD is acceptable.")
        sys.exit(0)

    # mode = args.mode

    # TO-DO: Understand the parameters and how the model loading works
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    net = IECore().read_network(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    exec_net = IECore().load_network(network=net, device_name=args.device)
    inputs = net.inputs["image"]

    h = inputs.shape[2] #368
    w = inputs.shape[3] #432
    print(h, w)

    threshold = 0.3
    nPoints = 18

    # TO-DO: Implement dynamic matrix with person tracking
    old_neck = -1*np.ones(20, dtype=int)
    new_neck = -1*np.ones(20, dtype=int)
    subject_height = -1*np.ones(20, dtype=int)
    fall_ratio = 0.5
    fallcount = 0
    
    try:
        while True:
            t1 = time.perf_counter()

            ret, color_image = cap.read()
            if not ret:
                break

            # TO-DO: Resize output screen?
            colw = color_image.shape[1]
            colh = color_image.shape[0]
            new_w = int(colw * min(w/colw, h/colh))
            new_h = int(colh * min(w/colw, h/colh))

            resized_image = cv2.resize(color_image, (w, new_h), interpolation = cv2.INTER_NEAREST)
            canvas = np.full((h, w, 3), 128)
            canvas[(h - new_h)//2:(h - new_h)//2 + new_h,(w - new_w)//2:(w - new_w)//2 + new_w, :] = resized_image

            prepimg = canvas
            prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
            prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW, (1, 3, 368, 432)
            outputs = exec_net.infer(inputs={input_blob: prepimg})["Openpose/concat_stage7"]

            detected_keypoints = []
            keypoints_list = np.zeros((0, 3))
            keypoint_id = 0
            
            for part in range(nPoints):
                probMap = outputs[0, part, :, :]
                probMap = cv2.resize(probMap, (canvas.shape[1], canvas.shape[0])) # (432, 368)
                keypoints = getKeypoints()
                keypoints_with_id = []

                for i in range(len(keypoints)):
                    keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                    keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                    keypoint_id += 1

                detected_keypoints.append(keypoints_with_id)

            frameClone = np.uint8(canvas.copy())
            for i in range(nPoints):
                for j in range(len(detected_keypoints[i])):
                    cv2.circle(frameClone, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)

            valid_pairs, invalid_pairs = getValidPairs()
            personwiseKeypoints = getPersonwiseKeypoints()
            
            # Resets points if no one is detected in the frame to prevent false positives between frames
            if range(len(personwiseKeypoints)) == 0:
                old_neck = -1*np.ones(20, dtype=int)
                new_neck = -1*np.ones(20, dtype=int)
                subject_height = -1*np.ones(20, dtype=int)

            # Fall algorithm
            # TO-DO: Optimise loops (multithreading?)
            for n in range(len(personwiseKeypoints)):
                for i in range(18):
                    index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                    
                    if -1 in index:
                        continue
                    
                    B = np.int32(keypoints_list[index.astype(int), 0])
                    A = np.int32(keypoints_list[index.astype(int), 1])
                    cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
                    
                    # Detect falling from neck points
                    if i == 0:
                        new_neck[n] = A[0]  
                    if i == 8:
                        subject_height[n] = A[0]-new_neck[n]
                    
                if totalframecount != 0 and totalframecount % 10 == 0:
                    if ((new_neck[n]-old_neck[n]) > subject_height[n]*fall_ratio) and new_neck[n] != -1 and old_neck[n] != -1 and subject_height[n] != -1:
                        fallcount += 1
                        cv2.imwrite(args.output_dir+'/img/'+str(fallcount)+'.jpg', frameClone)
            
                    old_neck[n] = new_neck[n]

            if fallcount != 0:
                cv2.putText(frameClone, "FALL COUNT: {0}".format(fallcount), (w-170,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
                
            cv2.putText(frameClone, fps, (w-170,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
            out.write(frameClone)
            cv2.namedWindow("USB Camera", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("USB Camera", frameClone)
            out.write(frameClone)

            if cv2.waitKey(1)&0xFF == ord('q'):
                break

            # FPS calculation
            framecount += 1
            totalframecount += 1
            if framecount >= 15:
                fps = "(Playback) {:.1f} FPS".format(time1/15)
                framecount = 0
                time1 = 0
            t2 = time.perf_counter()
            elapsedTime = t2-t1
            time1 += 1/elapsedTime

    except:
        import traceback
        traceback.print_exc()

    finally:
        cv2.destroyAllWindows()
        cap.release()
        out.release()
        print("\n\nFinished\n\n")
    
