from djitellopy import Tello
import cv2
import time

tello = Tello()
tello.connect()
tello.streamon()

i = 0
while True:
    frame = tello.get_frame_read().frame
    cv2.imshow("Tello", frame)

    key = cv2.waitKey(1)
    if key == ord("s"):
        cv2.imwrite(f"./calib/frame_{i}.jpg", frame)
        print("saved frame", i)
        i += 1

    if key == 27:  # ESC
        break

tello.streamoff()
cv2.destroyAllWindows()
