import cv2
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

# Инициализация видеозахвата
cap = cv2.VideoCapture("templates/istockphoto-1307407602-640_adpp_is.mp4")
backSub = cv2.createBackgroundSubtractorMOG2()

# Проверка успешности открытия файла
if not cap.isOpened():
    print("Ошибка при открытии видеофайла")
    exit()

# Создаем окно для отображения
cv2.namedWindow('Frame_final', cv2.WINDOW_NORMAL)

try:
    while cap.isOpened():
        # Чтение кадра
        ret, frame = cap.read()

        if not ret:
            print("Конец видео достигнут")
            break

        # Вычитание фона
        fg_mask = backSub.apply(frame)

        # Поиск контуров
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Рисование контуров на кадре
        frame_ct = cv2.drawContours(frame.copy(), contours, -1, (0, 255, 0), 2)

        # устанавливаем глобальный порог для удаления теней
        retval, mask_thresh = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)

        # вычисление ядра
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Apply erosion
        mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

        min_contour_area = 500  # Define your minimum area threshold
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        # Отображение результата

        frame_out = frame.copy()
        for cnt in large_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            frame_out = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 200), 3)

        # отображаем результат
        cv2.imshow('Frame_final', frame_out)
        cv2.imshow('fg_mask', fg_mask)

        # Обработка клавиши выхода
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

finally:
    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()