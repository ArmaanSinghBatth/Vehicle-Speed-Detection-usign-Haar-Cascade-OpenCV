import cv2
import time
import math

carCascade = cv2.CascadeClassifier('myhaar.xml')
video = cv2.VideoCapture('cars.mp4')

WIDTH = 1280
HEIGHT = 720

def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    ppm = 8.8  # pixels per meter (adjust based on calibration)
    d_meters = d_pixels / ppm
    fps = 18  # assume FPS of video
    speed = d_meters * fps * 3.6
    return speed

def trackMultipleObjects():
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0

    carTracker = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000

    while True:
        start_time = time.time()
        rc, image = video.read()
        if not rc:
            break

        image = cv2.resize(image, (WIDTH, HEIGHT))
        resultImage = image.copy()
        frameCounter += 1

        carIDtoDelete = []

        for carID in list(carTracker.keys()):
            success, bbox = carTracker[carID].update(image)
            if success:
                t_x, t_y, t_w, t_h = [int(v) for v in bbox]
                carLocation2[carID] = [t_x, t_y, t_w, t_h]
            else:
                carIDtoDelete.append(carID)

        for carID in carIDtoDelete:
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)

        if frameCounter % 10 == 0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

            for (_x, _y, _w, _h) in cars:
                x, y, w, h = int(_x), int(_y), int(_w), int(_h)
                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h
                matchCarID = None

                for carID in carTracker.keys():
                    [t_x, t_y, t_w, t_h] = carLocation2.get(carID, [0, 0, 0, 0])
                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    if ((t_x <= x_bar <= (t_x + t_w)) and
                        (t_y <= y_bar <= (t_y + t_h)) and
                        (x <= t_x_bar <= (x + w)) and
                        (y <= t_y_bar <= (y + h))):
                        matchCarID = carID

                if matchCarID is None:
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(image, (x, y, w, h))
                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h]
                    currentCarID += 1

        for carID in carTracker.keys():
            if carID in carLocation2:
                [x, y, w, h] = carLocation2[carID]
                cv2.rectangle(resultImage, (x, y), (x + w, y + h), rectangleColor, 4)

        end_time = time.time()
        fps = 1.0 / (end_time - start_time) if end_time != start_time else 0

        for i in carLocation1.keys():
            if i in carLocation2:
                [x1, y1, w1, h1] = carLocation1[i]
                [x2, y2, w2, h2] = carLocation2[i]
                carLocation1[i] = [x2, y2, w2, h2]

                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    if (speed[i] is None or speed[i] == 0) and 275 <= y1 <= 285:
                        speed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])

                    if speed[i] is not None and y1 >= 180:
                        cv2.putText(resultImage, str(int(speed[i])) + " km/hr",
                                    (int(x1 + w1 / 2), int(y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                    (255, 255, 255), 2)

        cv2.imshow('result', resultImage)
        if cv2.waitKey(33) == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    trackMultipleObjects()
