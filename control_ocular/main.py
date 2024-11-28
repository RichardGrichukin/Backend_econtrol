from tkinter import Tk, Button
from threading import Thread
from face_scan import scan_face
from eye_tracking import calibrate_cursor, track_eyes_and_blinks
import cv2

# Variables globales
cap = None  # Cámara compartida entre funciones

def iniciar_seguimiento():
    """Calibra el cursor y luego inicia el seguimiento ocular."""
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)  # Abre la cámara si no está abierta

    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    # Ejecutar la calibración
    calibrate_cursor(cap)

    # Iniciar el seguimiento ocular en un hilo separado
    seguimiento_hilo = Thread(target=track_eyes_and_blinks, args=(cap,))
    seguimiento_hilo.start()

def detener():
    """Detiene el seguimiento ocular."""
    global cap
    if cap and cap.isOpened():
        cap.release()  # Libera la cámara
        print("Seguimiento detenido.")

# Interfaz gráfica
root = Tk()
root.title("Control Ocular")
root.geometry("300x200")

Button(root, text="Registrar Rostro", command=scan_face).pack(pady=10)
Button(root, text="Iniciar Seguimiento", command=iniciar_seguimiento).pack(pady=10)
Button(root, text="Detener Seguimiento", command=detener).pack(pady=10)

# Cerrar correctamente la aplicación
def on_close():
    detener()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)

root.mainloop()

