import cv2
import numpy as np

def find_road_number(image: np.ndarray) -> int:
    """
    Найти номер дороги, на которой нет препятсвия в конце пути.

    :param image: исходное изображение
    :return: номер дороги, на котором нет препятсвия на дороге
    """

    h, w = image.shape[:2]
    road_number = -1
    car_road_number = -1
    prev_c = None
    flag = False
    for i in range(w):
        for j in range(h):
            b, g, r = image[j][i]
            if (b,g,r) == (213,213,213):
                if prev_c == "y":
                    road_number += 1
                    flag = False
                prev_c = "g"

            elif (128 <= b <= 255) and (0 <= r <= 128) and (0 <= g <= 255):
                car_road_number = road_number
                prev_c = "b"

            elif 121 <= int(b) <= 178 and 243 <= int(g) <= 252 and 244 <= int(r) <= 255:
                if prev_c == "g":
                    if not flag:
                        return road_number
                prev_c = "y"

            elif 0 <= b <= 128 and 128 <= r <= 255 and 0 <= g <= 128:
                flag = True
                break

    if road_number == car_road_number:
        return "не нужно перестраиваться"

    return road_number
