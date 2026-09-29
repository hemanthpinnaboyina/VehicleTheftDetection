
import cv2


class IoTCamera:

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.camera = None

    def start(self):
        self.camera = cv2.VideoCapture(self.camera_index)

        if not self.camera.isOpened():
            raise RuntimeError("Unable to open camera")

    def read_frame(self):
        if self.camera is None:
            return False, None

        success, frame = self.camera.read()

        return success, frame

    def stop(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None


def detect_vehicles(model, frame, confidence=0.35):

    results = model(
        frame,
        conf=confidence,
        verbose=False
    )

    result = results[0]

    annotated_frame = result.plot()

    detected_vehicles = []

    vehicle_classes = {
        "car",
        "motorcycle",
        "bus",
        "truck"
    }

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])

            class_name = model.names[class_id]

            score = float(box.conf[0])

            if class_name in vehicle_classes:

                detected_vehicles.append({
                    "name": class_name,
                    "confidence": score
                })

    vehicle_count = len(detected_vehicles)

    return annotated_frame, detected_vehicles