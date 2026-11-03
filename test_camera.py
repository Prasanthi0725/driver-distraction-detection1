
import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit(1)

print("Webcam opened. Press Q to close.")

while True:
    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read webcam frame.")
        break

    cv2.putText(
        frame,
        "Driver Monitoring - Camera Test",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Driver Distraction System - Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
