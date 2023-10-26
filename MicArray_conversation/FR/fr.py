import face_recognition
import cv2
import math


def run(known_face_images, known_face_names):
    # Load known face images and their names
    #known_face_images = ["person1.jpg", "person2.jpg", "person3.jpg"]
    #known_face_names = ["Person 1", "Person 2", "Person 3"]

    #known_face_images = ["known_face.jpg"]
    #known_face_names = ["known_person_01"]


    # Initialize lists to store known face encodings
    known_face_encodings = []

    # Load and encode the known faces
    for image_file in known_face_images:
        image = face_recognition.load_image_file(image_file)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)

    # Initialize the webcam
    '''
    for camera_idx in range(10):
        video_capture = cv2.VideoCapture(camera_idx)
        if video_capture.isOpened():
            print(f'연결된 카메라가 있음: {camera_idx}')
            break
    '''
    camera_idx = 0
    #camera_idx = 1
    video_capture = cv2.VideoCapture(camera_idx)
    if not video_capture.isOpened():
        print(f'연결된 카메라가 없음: {camera_idx}')

    while True:
        # Capture a frame from the webcam
        ret, frame = video_capture.read()

        # Resize the frame for better performance
        #frame = cv2.resize(frame, (300, 300))


        # Find all face locations in the current frame
        face_locations = face_recognition.face_locations(frame)
        #face_locations = face_recognition.api.batch_face_locations(frame)

        
        # Encode the faces in the current frame
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any of the known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown Person"

            for i, match in enumerate(matches):
                if match:
                    name = known_face_names[i]
                    break

            # Draw a rectangle around the face and label it
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)


            MAX_ANGLE = 45*math.pi/180
            tan_MAX_ANGLE = math.tan(MAX_ANGLE)

            width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            img_center_x = int((left + right)/2)
            img_center_y = int((top + bottom)/2)
            frame_center_x = width/2
            ratio = frame_center_x - img_center_x
            img_ang = math.atan(ratio/frame_center_x*tan_MAX_ANGLE)

            cv2.circle(frame, (img_center_x, img_center_y), 10, (0,255,0),5) 

            print(f'center: {frame_center_x}, width: {width}')
            print(f'ratio: {ratio}')      
            print(f'img_ang: {img_ang*180/math.pi}')


        # Display the resulting image
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    known_face_images = ["known_face.jpg"]
    known_face_names = ["known_person_01"]

    run(known_face_images, known_face_names)