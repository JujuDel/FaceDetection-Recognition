# -*- coding: utf-8 -*-

import dlib
import cv2
import numpy as np


###############################################################################
#                                                                             #
#                                  DISPLAYS                                   #
#                                                                             #
###############################################################################


def drawPolyline(im, landmarks, start, end, isClosed=False):
    points = []
    for i in range(start, end+1):
        point = [landmarks.part(i).x, landmarks.part(i).y]
        points.append(point)

    points = np.array(points, dtype=np.int32)
    cv2.polylines(
        im, [points], isClosed, (255, 200, 0), thickness=2, lineType=cv2.LINE_8
        )


def renderFaceLines(im, landmarks, color=(0, 255, 0), radius=2):
    assert(landmarks.num_parts == 68)
    drawPolyline(im, landmarks, 0, 16)           # Jaw line
    drawPolyline(im, landmarks, 17, 21)          # Left eyebrow
    drawPolyline(im, landmarks, 22, 26)          # Right eyebrow
    drawPolyline(im, landmarks, 27, 30)          # Nose bridge
    drawPolyline(im, landmarks, 30, 35, True)    # Lower nose
    drawPolyline(im, landmarks, 36, 41, True)    # Left eye
    drawPolyline(im, landmarks, 42, 47, True)    # Right Eye
    drawPolyline(im, landmarks, 48, 59, True)    # Outer lip
    drawPolyline(im, landmarks, 60, 67, True)    # Inner lip


def renderFacePoints(im, landmarks, color=(0, 255, 0), radius=3):
    for p in landmarks.parts():
        cv2.circle(im, (p.x, p.y), radius, color, -1)


def writeLandmarksToFile(landmarks, landmarksFileName):
    with open(landmarksFileName, 'w') as f:
        for p in landmarks.parts():
            f.write("%s %s\n" % (int(p.x), int(p.y)))
        f.close()


###############################################################################
#                                                                             #
#                                    MAIN                                     #
#                                                                             #
###############################################################################


def main():
    # Landmark model location
    PREDICTOR_5_PATH = "./../data/models/shape_predictor_5_face_landmarks.dat"
    PREDICTOR_68_PATH = "./../data/models/shape_predictor_68_face_landmarks.dat"
    # Get the face detector
    faceDetector = dlib.get_frontal_face_detector()
    # The landmark detector is implemented in the shape_predictor class
    landmarkDetector_5 = dlib.shape_predictor(PREDICTOR_5_PATH)
    landmarkDetector_68 = dlib.shape_predictor(PREDICTOR_68_PATH)

    # Read image
    imageFilename = "./../data/images/faces/family.jpg"
#    imageFilename = "./../data/images/faces/multipleFaces.jpg"
#    imageFilename = "./../data/images/faces/moreFaces.jpg"
#    imageFilename = "./../data/images/faces/evenMoreFaces.jpg"
    im1 = cv2.imread(imageFilename)
    im2 = cv2.imread(imageFilename)
    imDlib = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    # landmarks will be stored in results/family_i.txt
    landmarksBasename = "results/family"

    # Detect faces in the image
    faceRects = faceDetector(imDlib, 1)
    print("Number of faces detected: ", len(faceRects))

    # Loop over all detected face rectangles
    for i in range(0, len(faceRects)):
        newRect = dlib.rectangle(
                int(faceRects[i].left()), int(faceRects[i].top()),
                int(faceRects[i].right()), int(faceRects[i].bottom()))

        # For every face rectangle, run landmarkDetector
        landmarks_5 = landmarkDetector_5(imDlib, newRect)
        landmarks_68 = landmarkDetector_68(imDlib, newRect)
        # Print number of landmarks
        if i == 0:
            print("Number of landmarks", len(landmarks_5.parts()),
                  "and", len(landmarks_68.parts()))

        # Draw landmarks on face
        renderFaceLines(im1, landmarks_68)
        renderFacePoints(im2, landmarks_5)

        landmarksFileName = landmarksBasename + "_" + str(i) + ".txt"
        print("Saving landmarks to", landmarksFileName)
        # Write landmarks to disk
        writeLandmarksToFile(landmarks_68, landmarksFileName)

    outputFileName_5 = "results/familyLandmarks_5.jpg"
    outputFileName_68 = "results/familyLandmarks_68.jpg"
    print("Saving output image to", outputFileName_5, "and", outputFileName_68)
    cv2.imwrite(outputFileName_68, im1)
    cv2.imwrite(outputFileName_5, im2)

    cv2.imshow("Facial Landmark detector: 68 points", im1)
    cv2.imshow("Facial Landmark detector: 5 points", im2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
