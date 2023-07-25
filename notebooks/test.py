import os
import numpy as np
import cv2

import defogger

data_path = os.path.join('..', 'data', 'video')

capture = cv2.VideoCapture(os.path.join(data_path, 'foggy_dashcam1.mp4'))

FPS = capture.get(cv2.CAP_PROP_FPS)
HEIGHT = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
WIDTH = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
COUNT = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

for frame_id in range(COUNT):
    ret, frame = capture.read()
    if ret:
        frame = cv2.resize(frame, (WIDTH//3, HEIGHT//3))[:-HEIGHT//9, :, :]
        frame_corr, hazemap = defogger.remove_haze(frame, showHazeTransmissionMap = False)
        # frame_corr = defogger.remove_haze(frame, showHazeTransmissionMap = False)
        cv2.imshow("Dashcam Video", frame_corr)
    else:
        print("ERROR")
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
capture.release()