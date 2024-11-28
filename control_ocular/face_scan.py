import os
import cv2
import mediapipe as mp

# Crear el directorio assets si no existe
if not os.path.exists("assets"):
    os.makedirs("assets")

# Configuración de Mediapipe
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def scan_face():
    print("Iniciando escaneo de rostro...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Usar DirectShow en lugar de MSMF
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Establecer ancho
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Establecer alto

    if not cap.isOpened():
        print("No se pudo abrir la cámara. Verifica tu dispositivo.")
        return

    print("Cámara abierta correctamente. Iniciando detección de rostros...")
    user_data = None

    while cap.isOpened():
        print("Leyendo frame...")
        success, image = cap.read()
        if not success:
            print("No se pudo capturar un frame. Finalizando...")
            break

        # Procesar imagen
        print("Procesando frame...")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        if results.detections:
            print("Rostros detectados...")
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                x_min, y_min = int(bbox.xmin * w), int(bbox.ymin * h)
                width, height = int(bbox.width * w), int(bbox.height * h)
                user_data = image[y_min:y_min + height, x_min:x_min + width]
                cv2.rectangle(image, (x_min, y_min), (x_min + width, y_min + height), (0, 255, 0), 2)

        cv2.imshow('Scan Face', image)
        if cv2.waitKey(1) & 0xFF == 27:  # Presionar ESC para salir
            print("Saliendo...")
            break

    cap.release()
    cv2.destroyAllWindows()

    if user_data is not None:
        print("Guardando rostro...")
        cv2.imwrite("assets/user_face.jpg", user_data)
        print("Rostro guardado exitosamente en 'assets/user_face.jpg'")
    else:
        print("No se detectó ningún rostro.")

if __name__ == "__main__":
    scan_face()

