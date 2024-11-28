import cv2
import mediapipe as mp
import pyautogui
import time
import math

# Configuración de Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# IDs de puntos clave en los ojos
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Ajustes
AMPLIFY_FACTOR = 5.0  # Sensibilidad de movimiento
BLINK_THRESHOLD = 0.2  # Umbral para detectar parpadeos
BLINK_DURATION_THRESHOLD = 0.2  # Duración mínima para un parpadeo intencional (segundos)
CURSOR_UPDATE_DELAY = 0.05  # Retraso entre actualizaciones del cursor (segundos)
CLICK_DELAY = 0.7  # Tiempo mínimo entre clics (segundos)
pyautogui.FAILSAFE = False

# Variables globales
calibration_data = {}
last_cursor_update_time = 0
last_left_blink_time = 0
last_right_blink_time = 0
left_eye_closed_start = None
right_eye_closed_start = None


def euclidean_distance(point1, point2):
    """Calcula la distancia euclidiana entre dos puntos."""
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def calculate_ear(eye_landmarks, face_landmarks):
    """Calcula el EAR (Eye Aspect Ratio) para un ojo."""
    # Distancias verticales
    vertical_1 = euclidean_distance(face_landmarks.landmark[eye_landmarks[1]],
                                    face_landmarks.landmark[eye_landmarks[5]])
    vertical_2 = euclidean_distance(face_landmarks.landmark[eye_landmarks[2]],
                                    face_landmarks.landmark[eye_landmarks[4]])

    # Distancia horizontal
    horizontal = euclidean_distance(face_landmarks.landmark[eye_landmarks[0]],
                                     face_landmarks.landmark[eye_landmarks[3]])

    # EAR
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def calibrate_cursor(cap):
    """Realiza la calibración del cursor."""
    global calibration_data

    print("Iniciando calibración del cursor...")
    screen_width, screen_height = pyautogui.size()
    positions = {
        "centro": (screen_width // 2, screen_height // 2),
        "arriba izquierda": (screen_width // 4, screen_height // 4),
        "arriba derecha": (screen_width * 3 // 4, screen_height // 4),
        "abajo izquierda": (screen_width // 4, screen_height * 3 // 4),
        "abajo derecha": (screen_width * 3 // 4, screen_height * 3 // 4)
    }

    for label, pos in positions.items():
        pyautogui.moveTo(*pos)
        print(f"Mira al punto {label} en la pantalla...")
        time.sleep(1)

        # Registrar coordenadas
        start_time = time.time()
        samples_x, samples_y = [], []

        while time.time() - start_time < 2:  # Recolectar datos por 2 segundos
            success, image = cap.read()
            if not success:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    left_eye_x = sum([face_landmarks.landmark[i].x for i in LEFT_EYE]) / len(LEFT_EYE)
                    left_eye_y = sum([face_landmarks.landmark[i].y for i in LEFT_EYE]) / len(LEFT_EYE)
                    right_eye_x = sum([face_landmarks.landmark[i].x for i in RIGHT_EYE]) / len(RIGHT_EYE)
                    right_eye_y = sum([face_landmarks.landmark[i].y for i in RIGHT_EYE]) / len(RIGHT_EYE)

                    center_x = (left_eye_x + right_eye_x) / 2
                    center_y = (left_eye_y + right_eye_y) / 2

                    if 0 <= center_x <= 1 and 0 <= center_y <= 1:
                        samples_x.append(center_x)
                        samples_y.append(center_y)

        # Promediar los datos
        avg_x = sum(samples_x) / len(samples_x) if samples_x else 0.5
        avg_y = sum(samples_y) / len(samples_y) if samples_y else 0.5
        calibration_data[label] = (avg_x, avg_y)
        print(f"Calibración completada para {label}: ({avg_x}, {avg_y})")

    print("Calibración finalizada.")


def map_eye_to_screen(eye_x, eye_y):
    """Mapea las coordenadas del ojo a la pantalla."""
    dx = -(eye_x - calibration_data["centro"][0])  # Invertir desplazamiento horizontal
    dy = eye_y - calibration_data["centro"][1]  # Mantener desplazamiento vertical

    cursor_x = pyautogui.size().width // 2 + dx * pyautogui.size().width * AMPLIFY_FACTOR
    cursor_y = pyautogui.size().height // 2 + dy * pyautogui.size().height * AMPLIFY_FACTOR

    cursor_x = max(0, min(pyautogui.size().width, int(cursor_x)))
    cursor_y = max(0, min(pyautogui.size().height, int(cursor_y)))

    return cursor_x, cursor_y


def track_eyes_and_blinks(cap):
    """Rastrea los ojos y detecta parpadeos."""
    global last_cursor_update_time, last_left_blink_time, last_right_blink_time
    global left_eye_closed_start, right_eye_closed_start

    print("Iniciando seguimiento ocular y detección de parpadeos...")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("No se pudo capturar un frame. Finalizando...")
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Calcular EAR para ambos ojos
                left_ear = calculate_ear(LEFT_EYE, face_landmarks)
                right_ear = calculate_ear(RIGHT_EYE, face_landmarks)

                # Actualizar el cursor
                current_time = time.time()
                if current_time - last_cursor_update_time > CURSOR_UPDATE_DELAY:
                    left_eye_x = sum([face_landmarks.landmark[i].x for i in LEFT_EYE]) / len(LEFT_EYE)
                    left_eye_y = sum([face_landmarks.landmark[i].y for i in LEFT_EYE]) / len(LEFT_EYE)
                    right_eye_x = sum([face_landmarks.landmark[i].x for i in RIGHT_EYE]) / len(RIGHT_EYE)
                    right_eye_y = sum([face_landmarks.landmark[i].y for i in RIGHT_EYE]) / len(RIGHT_EYE)

                    eye_x = (left_eye_x + right_eye_x) / 2
                    eye_y = (left_eye_y + right_eye_y) / 2

                    cursor_x, cursor_y = map_eye_to_screen(eye_x, eye_y)
                    pyautogui.moveTo(cursor_x, cursor_y)
                    last_cursor_update_time = current_time

                # Detectar parpadeos
                if left_ear < BLINK_THRESHOLD and right_ear > BLINK_THRESHOLD:
                    if current_time - last_left_blink_time > CLICK_DELAY:
                        pyautogui.click(button='left')
                        last_left_blink_time = current_time

                if right_ear < BLINK_THRESHOLD and left_ear > BLINK_THRESHOLD:
                    if current_time - last_right_blink_time > CLICK_DELAY:
                        pyautogui.click(button='right')
                        last_right_blink_time = current_time

        if cv2.waitKey(1) & 0xFF == 27:
            break


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    calibrate_cursor(cap)  # Calibración inicial
    track_eyes_and_blinks(cap)  # Rastreo y parpadeos

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

