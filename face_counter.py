import cv2
import numpy as np
import time

try:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
except AttributeError:
    import os
    cascade_path = os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml")

face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise SystemExit("Could not load Haar cascade from: " + cascade_path)


class CentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=80):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.total_unique = 0

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1
        self.total_unique += 1

    def deregister(self, obj_id):
        del self.objects[obj_id]
        del self.disappeared[obj_id]

    def update(self, rects):
        if len(rects) == 0:
            for obj_id in list(self.disappeared):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        input_centroids = np.array([(x + w // 2, y + h // 2) for (x, y, w, h) in rects])

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(tuple(c))
            return self.objects

        obj_ids = list(self.objects.keys())
        obj_cents = np.array(list(self.objects.values()))

        D = np.linalg.norm(obj_cents[:, None] - input_centroids[None, :], axis=2)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            if D[r, c] > self.max_distance:
                continue
            obj_id = obj_ids[r]
            self.objects[obj_id] = tuple(input_centroids[c])
            self.disappeared[obj_id] = 0
            used_rows.add(r)
            used_cols.add(c)

        for r in set(range(len(obj_ids))) - used_rows:
            obj_id = obj_ids[r]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        for c in set(range(len(input_centroids))) - used_cols:
            self.register(tuple(input_centroids[c]))

        return self.objects


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Try VideoCapture(1) or close apps using the camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tracker = CentroidTracker()
    prev_time = time.time()
    frame_count = 0
    rects = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 2 == 0:
            scale = 0.5
            small = cv2.resize(frame, None, fx=scale, fy=scale)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))
            rects = [(int(x / scale), int(y / scale), int(w / scale), int(h / scale)) for (x, y, w, h) in faces]

        objects = tracker.update(rects)

        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for obj_id, (cx, cy) in objects.items():
            cv2.putText(frame, f"ID {obj_id}", (int(cx) - 20, int(cy) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)

        now = time.time()
        fps = 1 / max(now - prev_time, 1e-6)
        prev_time = now

        cv2.rectangle(frame, (0, 0), (260, 90), (0, 0, 0), -1)
        cv2.putText(frame, f"Current Faces: {len(rects)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Unique Total: {tracker.total_unique}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Real-Time Face Detection & Counting", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
