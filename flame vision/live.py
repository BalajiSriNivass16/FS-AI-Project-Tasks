from ultralytics import YOLO
import cv2
import winsound

model = YOLO("best.pt")
cap = cv2.VideoCapture("assets/fire_video.mp4")   # change to your video's exact name

if not cap.isOpened():
    print("Could not open the video. Check the file name and the assets folder.")
    raise SystemExit

while True:
    ok, frame = cap.read()
    if not ok:                      # video finished
        print("Video ended. No fire detected.")
        break

    results = model(frame, conf=0.5, verbose=False)
    annotated = results[0].plot()   # frame with detection boxes

    # Picture-in-picture: small copy of the original video in the top-right corner
    h, w = annotated.shape[:2]
    pip = cv2.resize(frame, (w // 4, h // 4))
    ph, pw = pip.shape[:2]
    annotated[10:10 + ph, w - pw - 10:w - 10] = pip
    cv2.rectangle(annotated, (w - pw - 10, 10), (w - 10, 10 + ph), (255, 255, 255), 2)

    cv2.imshow("Fire Detection", annotated)
    key = cv2.waitKey(1) & 0xFF     # lets the window refresh

    if len(results[0].boxes) > 0:
        print("FIRE DETECTED!")
        winsound.Beep(1000, 3000)   # 3 second beep
        break

    if key == ord("q"):             # press q to quit manually
        break

cap.release()
cv2.destroyAllWindows()