import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

def main():
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector(maxFaces=1)
    W = 6.3  # Object width in centimeters
    f = 500  # Focal length

    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img, draw=False)

        if faces:
            face = faces[0]
            pointLeft = face[145]
            pointRight = face[374]

            # Drawing
            cv2.line(img, pointLeft, pointRight, (0, 200, 0), 3)
            cv2.circle(img, pointLeft, 5, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, pointRight, 5, (255, 0, 255), cv2.FILLED)

            s, _ = detector.findDistance(pointLeft, pointRight)
            d = (W * f) / s  # Finding distance
            print(d)
            cvzone.putTextRect(img, f'Depth: {int(d)}cm', (face[10][0] - 100, face[10][1] - 50), scale=2)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()