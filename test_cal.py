import numpy as np
import cv2 as cv
import glob
import pickle
import os


def create_undistortion_maps(cameraMatrix, dist, frameSize):
    # Создаем оптимальную матрицу преобразования с сохранением всей области видимости
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(
        cameraMatrix,
        dist,
        frameSize,
        1,  # alpha=1 для максимального сохранения области видимости
        frameSize
    )

    # Создаем более точные карты преобразования с использованием 32-битных float
    mapx, mapy = cv.initUndistortRectifyMap(
        cameraMatrix,
        dist,
        None,
        newCameraMatrix,
        frameSize,
        cv.CV_32FC1  # Высокая точность тип данных
    )

    return mapx, mapy, newCameraMatrix, roi


def process_image(img, cameraMatrix, dist):
    # Получаем размеры изображения
    height, width = img.shape[:2]

    # Создаем карты преобразования
    mapx, mapy, newCameraMatrix, roi = create_undistortion_maps(
        cameraMatrix,
        dist,
        (width, height)
    )

    # Метод 1: Использование undistort с улучшенными параметрами
    dst1 = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

    # Метод 2: Использование remap с улучшенной интерполяцией
    dst2 = cv.remap(
        img,
        mapx,
        mapy,
        cv.INTER_CUBIC  # Более качественная интерполяция
    )

    # Обрезаем оба результата
    x, y, w, h = roi
    result1 = dst1[y:y + h, x:x + w]
    result2 = dst2[y:y + h, x:x + w]

    return result1, result2


def main():
    # Настройки шахматной доски
    chessboardSize = (6, 9)
    frameSize = (640, 480)

    # Критерии остановки поиска углов
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Создаем массив для 3D точек
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

    # Размер квадрата шахматной доски в мм
    size_of_chessboard_squares_mm = 20
    objp = objp * size_of_chessboard_squares_mm

    # Массивы для хранения точек
    objpoints = []  # 3D точки в реальном мире
    imgpoints = []  # 2D точки в плоскости изображения

    # Поиск изображений
    images = glob.glob('./images/*.png')

    # Обработка изображений
    for image_path in images:
        # Чтение изображения
        img = cv.imread(image_path)
        if img is None:
            print(f"Ошибка: Не удалось загрузить изображение {image_path}")
            continue

        # Преобразование в grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Поиск углов шахматной доски
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

        if ret == True:
            # Добавление 3D точек
            objpoints.append(objp)

            # Уточнение координат углов
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Визуализация углов
            cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(1000)

    cv.destroyAllWindows()

    # Калибровка камеры
    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints,
        imgpoints,
        frameSize,
        None,
        None
    )

    # Сохранение результатов калибровки
    pickle.dump((cameraMatrix, dist), open("calibration.pkl", "wb"))
    pickle.dump(cameraMatrix, open("cameraMatrix.pkl", "wb"))
    pickle.dump(dist, open("dist.pkl", "wb"))

    # Обработка тестового изображения
    test_image_path = 'cali5.png'
    if not os.path.exists(test_image_path):
        print(f"Ошибка: Файл '{test_image_path}' не найден")
        return

    # Загрузка и обработка тестового изображения
    img = cv.imread(test_image_path)
    if img is None:
        print(f"Ошибка: Не удалось загрузить тестовое изображение")
        return

    # Получение улучшенных результатов
    result1, result2 = process_image(img, cameraMatrix, dist)

    # Сохранение результатов с максимальным качеством
    cv.imwrite('caliResult1.png', result1, [int(cv.IMWRITE_PNG_COMPRESSION), 9])
    cv.imwrite('caliResult2.png', result2, [int(cv.IMWRITE_PNG_COMPRESSION), 9])

    # Расчет средней ошибки репроекции
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error

    print(f"Средняя ошибка репроекции: {mean_error / len(objpoints)}")


if __name__ == "__main__":
    main()