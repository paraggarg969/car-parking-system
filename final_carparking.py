# import cv2
# import pickle
#
# import numpy as np
#
# width = 107
# height = 48
#
# with open('carparkpos', 'rb') as f:
#     posList = pickle.load(f)
#
# cap = cv2.VideoCapture("Image_Video/video.mp4")
#
# # count non-zero pixels in each of the parking slot
# def checkparkingspace(preprocessed_frame):
#     counter = 0
#     vacant_spot_ids = []
#
#     if len(posList)!=0:
#         for i, pos in enumerate(posList):
#             x, y = pos
#             cropped_frame = preprocessed_frame[y:y+height, x:x+width]
#             # it give all 71 frame individual
#             # cv2.imshow(str(x*y), cropped_frame)
#
#             count = cv2.countNonZero(cropped_frame)
#
#             if count<900:
#                 counter+=1
#                 vacant_spot_ids.append(i + 1)
#                 # change the color of vacant parking space
#                 color = (100,255,100)
#             else:
#                 color = (100,100,255)
#
#             cv2.rectangle(frame, (pos[0],pos[1]), (pos[0] + width, pos[1] + height), color, 2)
#             cv2.putText(frame, str(i + 1), (pos[0], pos[1]+5), 0 , 0.5, [255,255,255], thickness=1, lineType=cv2.LINE_AA)
#
#         cv2.rectangle(frame, (51,15), (51+width+100, 15+height+10), (255, 0, 255), cv2.FILLED)
#         cv2.putText(frame, f'Free: {counter}/{len(posList)}', (52, 15+height), 0, 1, [255,255,255], thickness=2, lineType=cv2.LINE_AA)
#
#         # Print the IDs of all vacant parking spots in the terminal
#         print("Vacant Spot IDs:", vacant_spot_ids)
#
# while True:
#     # video is short in length then we want to continue play this video, don't want to stop it
#     # position of current frame is equal to total frame count then
#     if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
#         # it start with again zero frame
#         cap.set(cv2.CAP_PROP_POS_FRAMES,0)
#
#     ret, frame = cap.read()
#     if ret:
#         # convert our frame to grayscale
#         gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#         # Add gaussian Blur
#         blur = cv2.GaussianBlur(gray_scale, (3,3), 1)
#
#         # Applying threshold on each of the frame of the video             and 25, 16 is depend on your video
#         frame_threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
#
#         # to remove the noise dots as will apply median blur
#         median_blur = cv2.medianBlur(frame_threshold, 5)
#
#         # Applying dilation to increase the thickness of our edges
#         kernel = np.ones((3,3), np.uint8)
#         frame_dilate = cv2.dilate(median_blur, kernel, iterations=1)
#
#         checkparkingspace(frame_dilate)
#
#         cv2.imshow("video", frame)
#         if cv2.waitKey(1) & 0xFF == ord('1'):
#             break
#     else:
#         break



from flask import Flask, jsonify
import cv2
import pickle
import numpy as np
import threading

app = Flask(__name__)

# Load parking positions
width = 107
height = 48
with open('carparkpos', 'rb') as f:
    posList = pickle.load(f)

cap = cv2.VideoCapture("Image_Video/video.mp4")
vacant_spot_ids = []  # Global list to store vacant spot IDs
total_spots = len(posList)  # Total parking spots

# Function to process video, display it, and update vacant spot IDs
def process_video():
    global vacant_spot_ids
    while True:
        # Loop the video if it reaches the end
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale and apply preprocessing
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray_scale, (3, 3), 1)
        frame_threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 25, 16)
        median_blur = cv2.medianBlur(frame_threshold, 5)
        kernel = np.ones((3, 3), np.uint8)
        frame_dilate = cv2.dilate(median_blur, kernel, iterations=1)

        # Update vacant spot IDs by checking each parking spot
        vacant_spot_ids = []  # Reset for each frame
        for i, pos in enumerate(posList):
            x, y = pos
            cropped_frame = frame_dilate[y:y+height, x:x+width]
            count = cv2.countNonZero(cropped_frame)

            # Determine spot color based on occupancy
            if count < 900:
                vacant_spot_ids.append(i + 1)  # Add ID of vacant spot
                color = (0, 255, 0)  # Green for vacant
            else:
                color = (0, 0, 255)  # Red for occupied

            # Draw the rectangle and spot ID on the frame
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            cv2.putText(frame, f"ID: {i + 1}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the updated video frame only once
        cv2.imshow("Parking Management", frame)

        # Print total capacity and available spaces in the console
        available_spaces = len(vacant_spot_ids)
        print(f"Total: {total_spots} | Available: {available_spaces}")

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Start video processing in a separate thread
video_thread = threading.Thread(target=process_video)
video_thread.daemon = True
video_thread.start()

# API endpoint to get vacant spot IDs
@app.route('/vacant_spots', methods=['GET'])
def get_vacant_spots():
    available_spaces = len(vacant_spot_ids)
    return jsonify({
        "total_capacity": total_spots,
        "available_spaces": available_spaces,
        "vacant_spot_ids": vacant_spot_ids
    })

if __name__ == '__main__':
    app.run(debug=True)
