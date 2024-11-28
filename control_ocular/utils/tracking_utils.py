def normalize_coordinates(x, y, screen_width, screen_height):
    return int(x * screen_width), int(y * screen_height)

def draw_landmarks(image, landmarks, connections, color=(0, 255, 0)):
    for connection in connections:
        start = landmarks[connection[0]]
        end = landmarks[connection[1]]
        cv2.line(image, start, end, color, 2)
