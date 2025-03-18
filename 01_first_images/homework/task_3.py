import cv2
import numpy as np
import math

def rotate(image, point: tuple, angle: float) -> np.ndarray:
    """
    Повернуть изображение по часовой стрелке на угол от 0 до 360 градусов и преобразовать размер изображения.

    :param image: исходное изображение
    :param point: значение точки (x, y), вокруг которой повернуть изображение
    :param angle: угол поворота
    :return: повернутное изображение
    """

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    M[0, 2] += new_w / 2  - w//2
    M[1, 2] += new_h / 2 - h//2


    rotated_image = cv2.warpAffine(image.copy(), M, (new_w, new_h))

    return rotated_image


def apply_warpAffine(image, points1, points2) -> np.ndarray:
    """
    Применить афинное преобразование согласно переходу точек points1 -> points2 и
    преобразовать размер изображения.

    :param image:
    :param points1:
    :param points2:
    :return: преобразованное изображение
    """

    M = cv2.getPerspectiveTransform(points1, points2)
    x = max([int(p[0]) for p in points2])
    y = max([int(p[1]) for p in points2])

    rotated_image = cv2.warpPerspective(image, M, (x,y))

    return rotated_image

def new_wh(lu, ld, ru, rd) -> tuple:
    h = int(math.sqrt((lu[0] - ld[0])**2 + (lu[1] - ld[1])**2))
    w = int(math.sqrt((rd[0] - ld[0])**2 + (rd[1] - ld[1])**2))
    return h, w
