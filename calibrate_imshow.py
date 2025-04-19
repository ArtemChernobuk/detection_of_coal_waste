import cv2
import pickle
import numpy as np
# Загрузка параметров калибровки
with open("calibration.pkl", "rb") as f:
    cameraMatrix, dist = pickle.load(f)

# Инициализация видеопотока с камеры (0 — индекс камеры по умолчанию)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка: Не удалось открыть камеру!")
    exit()

# Получаем размер кадра для getOptimalNewCameraMatrix
ret, frame = cap.read()
if not ret:
    print("Ошибка: Не удалось получить кадр!")
    exit()

h, w = frame.shape[:2]
print(h, w)
frameSize = (w, h)

# Вычисляем оптимальную новую матрицу камеры и ROI
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
    cameraMatrix,
    dist,
    frameSize,
    1.0,  # alpha=1.0 — сохраняем максимальную область видимости
    frameSize,
    centerPrincipalPoint=True  # Центрирование оптического центра
)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: Не удалось получить кадр!")
        break

    # Коррекция дисторсии
    undistorted = cv2.undistort(frame, cameraMatrix, dist, None, newCameraMatrix)

    # Обрезка по ROI (если нужно)
    x, y, w, h = roi
    undistorted_cropped = undistorted[y:y+h, x:x+w]
    cropped_image = undistorted_cropped[0:85, 110:400]

    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

    # Определяем диапазоны цветов
    # Черный цвет имеет два диапазона в HSV пространстве
    lower_grey = np.array([0, 0, 50])
    upper_grey = np.array([180, 30, 200])

    mask_grey = cv2.inRange(hsv, lower_grey, upper_grey)

    contours_grey = cv2.findContours(mask_grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    for contour in contours_grey:
        if cv2.contourArea(contour) > 1000:  # Фильтруем шум
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(cropped_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            cv2.putText(frame, "red", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)

    print(cropped_image.shape)
    sections_col = 5
    sections = [cropped_image.shape[1] / sections_col]
    for i in range(sections_col):
        sections.append(sections[-1] + (cropped_image.shape[1] / sections_col))
    print(sections)


    for i in sections:
        cv2.line(cropped_image, (int(i), 0), (int(i), 85), (0, 0, 255), 5)
    # Вывод исходного и скорректированного изображения
    cv2.imshow('Cropped Image', cropped_image)
    cv2.imshow("Original", frame)
    cv2.imshow("Undistorted", undistorted_cropped)
    cv2.imshow("Black", mask_grey)

    # Выход по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()