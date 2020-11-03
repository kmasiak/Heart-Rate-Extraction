import cv2
import numpy as np
import dlib
from imutils import face_utils
import imutils
import face_recognition


class FaceDetection(object):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.fa = face_utils.FaceAligner(self.predictor, desiredFaceWidth=256)

    def face_detect(self, frame):

        # frame = imutils.resize(frame, width=400)
        face_frame = np.zeros((10, 10, 3), np.uint8)
        mask = np.zeros((10, 10, 3), np.uint8)
        ROI1 = np.zeros((10, 10, 3), np.uint8)
        ROI2 = np.zeros((10, 10, 3), np.uint8)
        # ROI3 = np.zeros((10, 10, 3), np.uint8)
        status = False

        if frame is None:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale image
        rects = self.detector(gray, 0)

        # loop over the face detections
        # for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array

        # assumpion: only 1 face is detected
        if len(rects) > 0:
            status = True
            # shape = self.predictor(gray, rects[0])
            # shape = face_utils.shape_to_np(shape)

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rects[0])
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            if y < 0:
                print("a")
                return frame, face_frame, ROI1, ROI2, status, mask
            # if i==0:
            face_frame = frame[y:y + h, x:x + w]
            # show the face number
            # cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
            #    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image

            # for (x, y) in shape:
            # cv2.circle(frame, (x, y), 1, (0, 0, 255), -1) #draw facial landmarks
            if (face_frame.shape[:2][1] != 0):
                face_frame = imutils.resize(face_frame, width=256)

            face_frame = self.fa.align(frame, gray, rects[0])  # align face

            grayf = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
            rectsf = self.detector(grayf, 0)

            if len(rectsf) > 0:
                shape = self.predictor(grayf, rectsf[0])
                shape = face_utils.shape_to_np(shape)

                for (a, b) in shape:
                    cv2.circle(face_frame, (a, b), 1, (0, 0, 255), -1)  # draw facial landmarks

                cv2.rectangle(face_frame, (shape[54][0], shape[29][1]),  # draw rectangle on right and left cheeks
                              (shape[12][0], shape[33][1]), (0, 255, 0), 0)
                cv2.rectangle(face_frame, (shape[4][0], shape[29][1]),
                              (shape[48][0], shape[33][1]), (0, 255, 0), 0)

                ROI1 = face_frame[shape[29][1]:shape[33][1],  # right cheek
                       shape[54][0]:shape[12][0]]

                ROI2 = face_frame[shape[29][1]:shape[33][1],  # left cheek
                       shape[4][0]:shape[48][0]]

                # ROI3 = face_frame[shape[29][1]:shape[33][1], #nose
                # shape[31][0]:shape[35][0]]

                # get the shape of face for color amplification
                rshape = np.zeros_like(shape)
                rshape = self.face_remap(shape)
                mask = np.zeros((face_frame.shape[0], face_frame.shape[1]))

                cv2.fillConvexPoly(mask, rshape[0:27], 1)
                # mask = np.zeros((face_frame.shape[0], face_frame.shape[1],3),np.uint8)
                # cv2.fillConvexPoly(mask, shape, 1)

            # cv2.imshow("face align", face_frame)

            # cv2.rectangle(frame,(shape[54][0], shape[29][1]), #draw rectangle on right and left cheeks
            # (shape[12][0],shape[54][1]), (0,255,0), 0)
            # cv2.rectangle(frame, (shape[4][0], shape[29][1]), 
            # (shape[48][0],shape[48][1]), (0,255,0), 0)

            # ROI1 = frame[shape[29][1]:shape[54][1], #right cheek
            # shape[54][0]:shape[12][0]]

            # ROI2 =  frame[shape[29][1]:shape[54][1], #left cheek
            # shape[4][0]:shape[48][0]]

        else:
            cv2.putText(frame, "No face detected",
                        (200, 200), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
            status = False

        lmm_image = face_recognition.load_image_file("test.jpg")
        lmm_face_encoding = face_recognition.face_encodings(lmm_image)[0]

        al_image = face_recognition.load_image_file("testImg.jpg")
        al_face_encoding = face_recognition.face_encodings(al_image)[0]

        known_faces = [
            lmm_face_encoding,
            al_face_encoding
        ]

        # Initialize some variables
        face_locations = []
        face_encodings = []
        face_names = []
        frame_number = 0

        #while True:
            # Grab a single frame of video
        #ret, frame = input_movie.read()
        #frame_number += 1
            # Quit when the input video file ends
        #if not ret:
        #    break
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]
           # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        print(face_encodings)
        face_names = []
        for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)
            print(match)
                # If you had more than 2 faces, you could make this logic a lot prettier
                # but I kept it simple for the demo
            name = None
            if match[0]:
                name = "Lin-Manuel Miranda"
            elif match[1]:
                name = "Face"

            face_names.append(name)
            im = []
            # Initialize
        im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Label the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            if not name:
                continue

            crop_img = frame[top:bottom, left:right]
            im_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                # print('cropped img', crop_img)

            from PIL import Image
            im = Image.fromarray(im_rgb)

            # im.save("myImg.jpeg")

            # Draw a box around the face
            # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            # cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            # font = cv2.FONT_HERSHEY_DUPLEX
            # cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

            # Write the resulting image to the output video file
            #print("Writing frame {} / {}".format(frame_number, length))
        # output_movie.write(frame)
        # crop_img = frame[top:bottom, left:right]
        # cv2.imshow("cropped", cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2))
        # top, right, bottom, left
        # gray = cv2.cvtColor(np.float32(im), cv2.COLOR_RGB2GRAY)
        # output_movie.write(im_rgb)
        # return im_rgb

        return frame, im_rgb, ROI1, ROI2, status, mask

        # some points in the facial landmarks need to be re-ordered

    def face_remap(self, shape):
        remapped_image = shape.copy()
        # left eye brow
        remapped_image[17] = shape[26]
        remapped_image[18] = shape[25]
        remapped_image[19] = shape[24]
        remapped_image[20] = shape[23]
        remapped_image[21] = shape[22]
        # right eye brow
        remapped_image[22] = shape[21]
        remapped_image[23] = shape[20]
        remapped_image[24] = shape[19]
        remapped_image[25] = shape[18]
        remapped_image[26] = shape[17]
        # neatening 
        remapped_image[27] = shape[0]

        remapped_image = cv2.convexHull(shape)
        return remapped_image
