import cv2
import numpy as np

# Инициализируем начальные значения параметров
lower_red1_h = 0
lower_red1_s = 100
lower_red1_v = 100
lower_green_h = 40
lower_green_s = 50
lower_green_v = 50


# Функции для обработки изменений слайдеров
def update_lower_red1_h(val):
    global lower_red1_h
    lower_red1_h = val


def update_lower_red1_s(val):
    global lower_red1_s
    lower_red1_s = val


def update_lower_red1_v(val):
    global lower_red1_v
    lower_red1_v = val


def update_lower_green_h(val):
    global lower_green_h
    lower_green_h = val


def update_lower_green_s(val):
    global lower_green_s
    lower_green_s = val


def update_lower_green_v(val):
    global lower_green_v
    lower_green_v = val


# Создаем окно для слайдеров
cv2.namedWindow('Settings')
cv2.resizeWindow('Settings', 800, 600)

# Создаем слайдеры для красного цвета
cv2.createTrackbar('H(red)', 'Settings', lower_red1_h, 179, update_lower_red1_h)
cv2.createTrackbar('S(red)', 'Settings', lower_red1_s, 255, update_lower_red1_s)
cv2.createTrackbar('V(red)', 'Settings', lower_red1_v, 255, update_lower_red1_v)

# Создаем слайдеры для зеленого цвета
cv2.createTrackbar('H(green)', 'Settings', lower_green_h, 179, update_lower_green_h)
cv2.createTrackbar('S(green)', 'Settings', lower_green_s, 255, update_lower_green_s)
cv2.createTrackbar('V(green)', 'Settings', lower_green_v, 255, update_lower_green_v)

# Инициализация камеры
cap = cv2.VideoCapture(1)

while True:
    # Получаем кадр с камеры
    ret, frame = cap.read()

    if not ret:
        break

    # Конвертируем кадр в формат HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Создаем массивы для текущих значений параметров
    lower_red1 = np.array([lower_red1_h, lower_red1_s, lower_red1_v])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    lower_green = np.array([lower_green_h, lower_green_s, lower_green_v])
    upper_green = np.array([80, 255, 255])

    # Создаем маски для красного цвета
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Создаем маску для зеленого цвета
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Находим контуры для красных объектов
    contours_red = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # Находим контуры для зеленых объектов
    contours_green = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # Рисуем прямоугольники и добавляем текстовые метки
    for contour in contours_red:
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Красный", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)

    for contour in contours_green:
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Зеленый", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

    # Отображаем результат
    cv2.imshow('Frame', frame)
    cv2.imshow('green', mask_green)
    cv2.imshow('red', mask_red)

    # Проверяем нажатие клавиши ESC для выхода
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()