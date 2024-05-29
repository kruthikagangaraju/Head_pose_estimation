import cv2
import mediapipe as mp
import numpy as np
import time
import datetime
import textwrap


# Function to get head pose estimation using MediaPipe class
def head_pose_detection(image, face_mesh, initial_angles, angles, diff_x, diff_y, diff_z):
    # Initialize the drawing utilities from MediaPipe and set the drawing specifications
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # Get the dimensions of the image
    img_h, img_w, img_c = image.shape
    face_3d = [] # List to store 3D coordinates of the face landmarks
    face_2d = [] # List to store 2D coordinates of the face landmarks

    # Process the image to detect face landmarks
    results = face_mesh.process(image)

    if results.multi_face_landmarks:
        # Loop through each face detected in the image
        for face_landmarks in results.multi_face_landmarks:
            # Loop through specific landmarks on the face
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [33, 263, 1, 61, 291, 199]: # Indices of landmarks to be used for pose estimation
                    if idx == 1:
                        # Special case for the nose landmark
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    # Convert landmark coordinates to image dimensions
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])
            # Convert lists to numpy arrays for further calculations
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                   [0, focal_length, img_h / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            # Solve the PnP problem to get rotation and translation vectors
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, jac = cv2.Rodrigues(rot_vec) # Convert rotation vector to rotation matrix
            if initial_angles is None:
                # If initial angles are not set, set them using the current rotation matrix
                initial_angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            else:
                # Otherwise, compute the current angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Calculate the initial and current angles in degrees
            init_x = int(round(initial_angles[0] * 360))
            init_y = int(round(initial_angles[1] * 360))
            init_z = int(round(initial_angles[2] * 360))
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            threshold = 6 # Threshold for detecting head movement
            # Determine the response based on the head movement
            if (abs(x - init_x) < threshold) and (abs(y - init_y) < threshold) and (abs(z - init_z) < threshold):
                text = " "
            elif (abs(x - init_x) > threshold) and (abs(y - init_y) < threshold) and (abs(z - init_z) < threshold):
                text = "Yes"
            elif (abs(x - init_x) < threshold) and (abs(y - init_y) > threshold) and (abs(z - init_z) < threshold):
                text = "No"
            else:
                text = "Invalid response; try again"
            # Append the differences to their respective lists
            diff_x = np.append(diff_x, int(round(abs(x - init_x))))
            diff_y = np.append(diff_y, int(round(abs(y - init_y))))
            diff_z = np.append(diff_z, int(round(abs(z - init_z))))

            # Add text overlays on the image to show instructions and some image details
            cv2.putText(image, "Press: 'g' to get question; 's' to show your answer;", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(image, "       'r' to reset answer; q' to quit", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(image, text, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            cv2.putText(image, "x:" + str(np.round(x, 2)), (520, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(image, "y:" + str(np.round(y, 2)), (520, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(image, "z:" + str(np.round(z, 2)), (520, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return image, initial_angles, angles, diff_x, diff_y, diff_z

def main():
    mp_face_mesh = mp.solutions.face_mesh # Import the face mesh solution module from MediaPipe
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) # Initialize the face mesh model with specified detection and tracking confidence levels
    cap = cv2.VideoCapture(0) # start video

    # Initialise variables
    initial_angles = None
    angles = np.array([0.0, 0.0, 0.0])
    diff_x = np.array([])
    diff_y = np.array([])
    diff_z = np.array([])
    x_sum = 0
    y_sum = 0
    q_arr = []
    a_arr = []
    t_arr = []

    # A predefined array of questions to be asked for user
    question_array = [
        " ",
        "Do you believe artificial intelligence will ultimately benefit society?",
        "Can we trust autonomous vehicles to make ethical decisions in life-threatening situations?",
        "Should we be concerned about the potential misuse of biometric data in facial recognition technology?",
        "Is the rapid advancement of virtual reality leading us towards a dystopian future of disconnection from reality?",
        "Do you think the widespread adoption of blockchain technology will revolutionize industries beyond finance?"
    ]

    question_index = 0
    answer = ""

    while cap.isOpened():
        success, image = cap.read() # reading the video
        if not success:
            break

        start = time.time()
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB) # Flip the image horizontally and convert the color space from BGR to RGB
        image.flags.writeable = False # Set the image as non-writeable to improve performance while processing
        image.flags.writeable = True # Set the image back to writeable after processing
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert the color space back from RGB to BGR

        # get the head pose and determine the response using the function
        image, initial_angles, angles, diff_x, diff_y, diff_z = head_pose_detection(image, face_mesh, initial_angles, angles, diff_x, diff_y, diff_z)
        # print(initial_angles)
        end = time.time()
        totalTime = end - start

        # Get the current fps and display on the video screen
        if totalTime > 0:
            fps = 1 / totalTime
            cv2.putText(image, f'FPS: {int(fps)}', (20, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(image, f'FPS: calculating...', (20, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, str(datetime.datetime.now()), (350, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        # print(str(datetime.datetime.now()))

        wrapped_text = textwrap.wrap(question_array[question_index], width=70)
        for i, line in enumerate(wrapped_text):
            cv2.putText(image, line, (10, 400 + (i * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) # Display current question
        if answer:
            cv2.putText(image, f"Answer: {answer}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) # Display the answer given by user

        cv2.imshow("Head Pose Estimation", image)

        # Sum the change in x and y values to determine final answer
        x_sum = np.sum(diff_x)
        y_sum = np.sum(diff_y)

        # Check the key pressed by user and perform necessary action
        key = cv2.waitKey(5) & 0xFF
        # Prints the next question for the user
        if key == ord('g'):
            question_index = (question_index + 1) % len(question_array)
            answer = ""
            initial_angles = None
            angles = np.array([0.0, 0.0, 0.0])
            diff_x = np.array([])
            diff_y = np.array([])
            diff_z = np.array([])
        # Displays the answer given by user for the displayed question
        elif key == ord('s'):
            if x_sum > y_sum:
                answer = "Yes"
            elif x_sum < y_sum:
                answer = "No"
            ans_timestamp = datetime.datetime.now() # Gets the timestamp of when the user answers the question
            q_arr.append(question_array[question_index])
            t_arr.append(ans_timestamp)
            a_arr.append(answer)
        # Resets the answer if the user is not satisfied with the displayed answer
        elif key == ord('r'):
            initial_angles = None
            angles = np.array([0.0, 0.0, 0.0])
            diff_x = np.array([])
            diff_y = np.array([])
            diff_z = np.array([])
            answer = ""
        # Quits the video
        elif key == ord('q'):
            break

    # print(diff_x, diff_y, diff_z)
    cap.release()
    cv2.destroyAllWindows()
    print("******* YOUR RESPONSES ********")
    for q, a, t in zip(q_arr, a_arr, t_arr):
        print(f"Question: {q}\nAnswer: {a}\nTimestamp: {t.strftime('%Y-%m-%d %H:%M:%S')}\n") # Displays all the questions, user given answers and the timestamp of the answer
    print("******* THANKS FOR PARTICIPATING! *******")


if __name__ == "__main__":
    main()
