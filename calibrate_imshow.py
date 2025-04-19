import cv2
import pickle

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

    cropped_image = undistorted_cropped[0:85, 110:440]

    cv2.imshow('Cropped Image', cropped_image)

    # Вывод исходного и скорректированного изображения
    cv2.imshow("Original", frame)
    cv2.imshow("Undistorted", undistorted_cropped)

    # Выход по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()