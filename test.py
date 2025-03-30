import cv2
import numpy as np

# Инициализация камеры
cap = cv2.VideoCapture(0)

while True:
    # Получаем кадр с камеры
    ret, frame = cap.read()

    if not ret:
        break

    # Конвертируем кадр в формат HSV для более удобной работы с цветами
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Определяем диапазоны цветов
    # Красный цвет имеет два диапазона в HSV пространстве
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    # Зеленый цвет
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])

    # Создаем маски для красного цвета
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Создаем маску для зеленого цвета
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Находим контуры для красных объектов
    contours_red = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    print(contours_red[-2])
    # Находим контуры для зеленых объектов
    contours_green = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    for contour in contours_red:
        if cv2.contourArea(contour) > 1000:  # Фильтруем шум
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # Добавляем текст "Красный" над прямоугольником
            cv2.putText(frame, "red", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)

    for contour in contours_green:
        if cv2.contourArea(contour) > 1000:  # Фильтруем шум
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Добавляем текст "Зеленый" над прямоугольником
            cv2.putText(frame, "green", (x, y - 10),
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