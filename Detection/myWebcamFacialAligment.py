# -*- coding: utf-8 -*-

import dlib
import cv2
import math
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


# Draw the lines for a 68 landmarks model
def renderFaceLines(im, landmarks, color=(0, 255, 0), radius=4):
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


# Draw points for any numbers of landmarks models
def renderFacePoints(im, landmarks, color=(0, 255, 0), radius=1):
    for p in landmarks.parts():
        cv2.circle(im, (p.x, p.y), radius, color, -1)


###############################################################################
#                                                                             #
#                                  LANDMARKS                                  #
#                                                                             #
###############################################################################


# convert Dlib shape detector object to list of tuples
def convertDlibLandmarksToPoints(dlibLandmarks):
    points = []
    for p in dlibLandmarks.parts():
        pt = (p.x, p.y)
        points.append(pt)
    return points


# Detect facial landmarks in image
def detectLandmarks(faceBox, landmarkDetector, im):
    rect = [faceBox.left(),
            faceBox.top(),
            faceBox.right(),
            faceBox.bottom()]
    rect = dlib.rectangle(*rect)

    landmarks = landmarkDetector(im, rect)
    points = convertDlibLandmarksToPoints(landmarks)

    return landmarks, points


###############################################################################
#                                                                             #
#                                FACE ALIGMENT                                #
#                                                                             #
###############################################################################


# Normalizes an align a facial image to a standard size given by outSize.
# After normalization:
#    - left corner of the left eye is at   ( 0.3 * w, h / 3 )
#    - right corner of the right eye is at ( 0.7 * w, h / 3 )
def alignImagesAndLandmarks(outSize, imIn, pointsIn):
    h, w = outSize

    # Corners of the eye in input image
    if len(pointsIn) == 68:   # 68 landmarks model
        eyecornerSrc = [pointsIn[36], pointsIn[45]]
    elif len(pointsIn) == 5:  # 5 landmarks model
        eyecornerSrc = [pointsIn[2], pointsIn[0]]
    else:                     # something else
        return None, []

    # Corners of the eye in normalized image
    eyecornerDst = [(np.int(0.3 * w), np.int(h/3)),
                    (np.int(0.7 * w), np.int(h/3))]

    # Calculate similarity transform
    tform = similarityTransform(eyecornerSrc, eyecornerDst)
    imOut = np.zeros(imIn.shape, dtype=imIn.dtype)

    # Apply similarity transform to input image
    imOut = cv2.warpAffine(imIn, tform, (w, h))

    # Apply similarity transform to landmarks
    points = np.reshape(pointsIn, (pointsIn.shape[0], 1, pointsIn.shape[1]))
    points = cv2.transform(points, tform)
    points = np.reshape(points, (pointsIn.shape[0], pointsIn.shape[1]))

    return imOut, points


# Compute the similarity transform
def similarityTransform(inPoints, outPoints):
    sin60 = math.sin(60 * math.pi / 180)
    cos60 = math.cos(60 * math.pi / 180)

    # Convert inPoints/outPoints to lists
    inPts = np.copy(inPoints).tolist()
    outPts = np.copy(outPoints).tolist()

    # For similarity calculus, 3 points are required:
    #  the third point is calculated so that the three points make
    #  an equilateral triangle.
    xin = inPts[1][0]
    xin = xin + cos60*(inPts[0][0] - inPts[1][0])
    xin = xin - sin60*(inPts[0][1] - inPts[1][1])

    yin = inPts[1][1]
    yin = yin + sin60*(inPts[0][0] - inPts[1][0])
    yin = yin + cos60*(inPts[0][1] - inPts[1][1])

    inPts.append([np.int(xin), np.int(yin)])

    xout = outPts[1][0]
    xout = xout + cos60*(outPts[0][0] - outPts[1][0])
    xout = xout - sin60*(outPts[0][1] - outPts[1][1])

    yout = outPts[1][1]
    yout = yout + sin60*(outPts[0][0] - outPts[1][0])
    yout = yout + cos60*(outPts[0][1] - outPts[1][1])

    outPts.append([np.int(xout), np.int(yout)])

    # Compute the similarity transform
    tform = cv2.estimateAffinePartial2D(np.array([inPts]), np.array([outPts]))
    return tform[0]


###############################################################################
#                                                                             #
#                                    MAIN                                     #
#                                                                             #
###############################################################################


def main():
    # Landmark model location
    PREDICTOR_68_PATH = "./data/models/shape_predictor_68_face_landmarks.dat"
    # Get the face detector
    faceDetector = dlib.get_frontal_face_detector()
    # The landmark detector is implemented in the shape_predictor class
    landmarkDetector_68 = dlib.shape_predictor(PREDICTOR_68_PATH)

    # Number of detected face in the previous frame
    nbFacesPrev = 0

    # Whether or not display the landmarks points
    displayPoints = False
    # Whether or not display the landmarks lines
    displayLines = False

    # Create a VideoCapture object
    video = cv2.VideoCapture(0)

    # Dimensions of output image
    h = 600
    w = 600

    while(True):
        # Grab a frame
        ret, frame = video.read()
        if not ret:
            break

        # BGR -> RGB
        imDlib = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect the faces
        faceRects = faceDetector(imDlib, 0)

        # Loop over all detected face
        for i in range(0, len(faceRects)):
            nameWindowFace = "Face " + str(i)

            # Detect the landmarks
            landmarks, points = detectLandmarks(
                faceRects[i],
                landmarkDetector_68,
                imDlib)

            # When landmarks are detected
            if len(points) > 0:
                # Draw landmarks on the detected face
                if displayPoints:
                    renderFacePoints(frame, landmarks)
                if displayLines:
                    renderFaceLines(frame, landmarks)

                # Convert the point list to a numpy array
                points = np.array(points)

                # Convert the image to floating point in the range 0 to 1
                frameNp = np.float32(frame) / 255.0

                # align the face to the output coordinates
                imNorm, points = alignImagesAndLandmarks(
                        (h, w), frameNp, points)
                if len(points) > 0:
                    # Go back to the range 0 to 255
                    imNorm = np.uint8(imNorm * 255)
                    # Display the face
                    cv2.imshow(nameWindowFace, imNorm)
                else:
                    # Destroy the window
                    cv2.destroyWindow(nameWindowFace)

        # Destroy the windows for which the tracking has been lost
        for i in range(len(faceRects), nbFacesPrev):
            cv2.destroyWindow("Face " + str(i))
        nbFacesPrev = len(faceRects)

        # Display the webcam
        cv2.putText(
            frame,
            "Display Lines ('L')",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
        cv2.putText(
            frame,
            "Display Lines ('P')",
            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
        cv2.imshow("Facial Landmark detectors", frame)

        # Key event
        key = cv2.waitKey(1)
        if key == 27:                             # ESC - quit the app
            break
        elif key == ord('p') or key == ord('P'):  # Display the landmarks point
            displayPoints = not displayPoints
        elif key == ord('l') or key == ord('L'):  # Display the landmarks lines
            displayLines = not displayLines

    # Destroy the windows
    cv2.destroyAllWindows()
    # Release the webcam
    video.release()

if __name__ == '__main__':
    main()
