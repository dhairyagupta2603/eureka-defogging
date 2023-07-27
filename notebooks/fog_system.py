import os
import cv2
import numpy as np

import ultralytics
print(ultralytics.checks())


data_path = os.path.join('..', 'data', 'video')
print(f'Data Path: {data_path}')

print('Initializing model...')
model = ultralytics.YOLO('yolov8n.pt')
print('Initialized model!')

def dehaze(image, fog_reduction=0.5):
    # Convert the image to float representation
    image = image.astype(np.float32) / 255.0

    # Estimate the atmospheric light
    dark_channel = np.min(image, axis=2)
    atmospheric_light = np.percentile(dark_channel, 99)

    # Estimate the transmission map using Dark Channel Prior
    transmission = 1 - fog_reduction * dark_channel / atmospheric_light

    # Clip transmission values to avoid artifacts
    transmission = np.clip(transmission, 0.1, 1.0)

    # Recover the scene radiance
    recovered_scene = np.zeros_like(image)
    for i in range(3):
        recovered_scene[:, :, i] = (image[:, :, i] - atmospheric_light) / transmission + atmospheric_light

    # Clip pixel values to the valid range
    recovered_scene = np.clip(recovered_scene, 0, 1)

    # Convert the image back to uint8 format
    recovered_scene = (recovered_scene * 255).astype(np.uint8)
    return recovered_scene


capture = cv2.VideoCapture(os.path.join(data_path, 'foggy_dashcam2.mp4'))
FPS = capture.get(cv2.CAP_PROP_FPS)
HEIGHT = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
WIDTH = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
COUNT = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

fog_reduction = 0.85  # Set the fog reduction level (0.0 to 1.0)

# Process each frame and write to the output video
for frame_id in range(COUNT):
    ret, frame = capture.read()
    if ret:
        frame = cv2.resize(frame, (WIDTH//4, HEIGHT//4))
        dehazed_frame = dehaze(frame, fog_reduction=fog_reduction)
        results = model(dehazed_frame)
        annotated_frame = results[0].plot()
        # out.write(dehazed_frame)
        cv2.imshow('dashcam', annotated_frame)
    else:
        print("ERROR")
        break
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
capture.release()
# out.release()
cv2.destroyAllWindows()