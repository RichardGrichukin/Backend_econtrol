import cv2

def open_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("No se pudo acceder a la cámara")
    return cap

def close_camera(cap):
    cap.release()
    cv2.destroyAllWindows()
