from django.shortcuts import render
import cv2
from .object_detector import HomogeneousBgDetector
import numpy as np
import io
from django.core.files.uploadedfile import InMemoryUploadedFile

def measure_object_size(request):
    object_measurements = []
    processed_image = None

    if request.method == 'POST' and 'image_file' in request.FILES:
        # Get the uploaded image file
        image_file = request.FILES['image_file']

        # Load the image using OpenCV
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Load Aruco detector
        parameters = cv2.aruco.DetectorParameters()
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

        # Load Object Detector
        detector = HomogeneousBgDetector()

        # Get Aruco marker
        corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

        # Draw polygon around the marker
        int_corners = np.int0(corners)
        cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

        # Aruco Perimeter
        aruco_perimeter = cv2.arcLength(corners[0], True)

        # Pixel to cm ratio
        pixel_cm_ratio = aruco_perimeter / 20

        contours = detector.detect_objects(img)

        # Draw objects boundaries
        for cnt in contours:
            # Get rect
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect

            # Get Width and Height of the Objects by applying the Ratio pixel to cm
            object_width = w / pixel_cm_ratio
            object_height = h / pixel_cm_ratio

            # Display rectangle
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.polylines(img, [box], True, (255, 0, 0), 2)
            cv2.putText(img, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
            cv2.putText(img, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

            object_measurements.append({
                'width': round(object_width, 1),
                'height': round(object_height, 1)
            })

        # Save the processed image to a buffer
        is_success, buffer = cv2.imencode(".jpg", img)
        io_buf = io.BytesIO(buffer)
        processed_image = InMemoryUploadedFile(io_buf, None, 'processed_image.jpg', 'image/jpeg', io_buf.getbuffer().nbytes, None)

    context = {'object_measurements': object_measurements, 'processed_image': processed_image}
    return render(request, 'object_measurement/measure_object_size.html', context)
