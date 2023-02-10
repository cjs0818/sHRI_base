import cv2

ESC = 27


if __name__ == '__main__':

    sysClosed = False

    while True:
        facial_status = ord('n')

        animation_file_normal = "FacialExpression/animation/0_normal.avi"
        animation_file_happy = "FacialExpression/animation/1_happy.avi"
        animation_file_surprise = "FacialExpression/animation/2_surprise.avi"
        animation_file_sad = "FacialExpression/animation/3_sad.avi"
        animation_file_angry = "FacialExpression/animation/4_angry.avi"

        cap_normal = cv2.VideoCapture(animation_file_normal)
        cap_happy = cv2.VideoCapture(animation_file_happy)
        cap_surprise = cv2.VideoCapture(animation_file_surprise)
        cap_sad = cv2.VideoCapture(animation_file_sad)
        cap_angry = cv2.VideoCapture(animation_file_angry)

        if not cap_normal.isOpened():
            raise IOError("Cannot Open the <Normal> Video")
        if not cap_happy.isOpened():
            raise IOError("Cannot Open the <Happy> Video")
        if not cap_surprise.isOpened():
            raise IOError("Cannot Open the <Surprise> Video")
        if not cap_sad.isOpened():
            raise IOError("Cannot Open the <Sad> Video")
        if not cap_angry.isOpened():
            raise IOError("Cannot Open the <Angry> Video")

        cv2.namedWindow('Facial Expression', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Facial Expression', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            input_value = cv2.waitKey(25)

            if input_value == ESC:
                sysClosed = True
                break

            if input_value == ord('n'):
                facial_status = ord('n')
            if input_value == ord('h'):
                facial_status = ord('h')
            if input_value == ord('r'):
                facial_status = ord('r')
            if input_value == ord('s'):
                facial_status = ord('s')
            if input_value == ord('a'):
                facial_status = ord('a')

            if facial_status == ord('n'):
                ret_normal, img_normal = cap_normal.read()
                if ret_normal == True:
                   cv2.imshow('Facial Expression', img_normal)
                else:
                    break
            elif facial_status == ord('h'):
                ret_happy, img_happy = cap_happy.read()
                if ret_happy == True:
                    cv2.imshow('Facial Expression', img_happy)
                else:
                    break
            elif facial_status == ord('r'):
                ret_surprise, img_surprise = cap_surprise.read()
                if ret_surprise == True:
                    cv2.imshow('Facial Expression', img_surprise)
                else:
                    break
            elif facial_status == ord('s'):
                ret_sad, img_sad = cap_sad.read()
                if ret_sad == True:
                    cv2.imshow('Facial Expression', img_sad)
                else:
                    break
            elif facial_status == ord('a'):
                ret_angry, img_angry = cap_angry.read()
                if ret_angry == True:
                    cv2.imshow('Facial Expression', img_angry)
                else:
                    break

        if sysClosed == True:
            break

    cap_normal.release()
    cap_happy.release()
    cap_surprise.release()
    cap_sad.release()
    cap_angry.release()
    cv2.destroyAllWindows()
