import cv2
import numpy as np
from enum import Enum


class Orientation(Enum):
    BOTTOM = 0
    TOP = 1
    LEFT = 2
    RIGHT = 3


WHITE_PIXEL = (255, 255, 255)
BLACK_PIXEL = (0, 0, 0)


def locate_maze_openings(maze_image: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Определяет координаты входа и выхода в лабиринте.

    :param maze_image: Изображение лабиринта в виде массива numpy.
    :return: Кортеж с координатами входа и выхода ((x_start, y_start), (x_end, y_end)).
    """
    height = maze_image.shape[0]
 
    start_white_indices = np.where(np.all(maze_image[0] == WHITE_PIXEL, axis=1))[0]
    start_x = int(start_white_indices[0]) if start_white_indices.size > 0 else 0

    end_white_indices = np.where(np.all(maze_image[height-1] == WHITE_PIXEL, axis=1))[0]
    end_x = int(end_white_indices[0]) if end_white_indices.size > 0 else 0

    if not start_x or not end_x:
        raise ValueError("Не удалось найти вход или выход в лабиринте.")

    return ((start_x, 0), (end_x, height - 1))


def determine_next_position(maze: np.ndarray, current: tuple[int, int], visited: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Определяет следующую позицию для перемещения по лабиринту.

    :param maze: Изображение лабиринта.
    :param current: Текущие координаты.
    :param visited: Матрица посещённых точек.
    :return: Кортеж с новой позицией и предыдущей позицией.
    """
    height= maze.shape[0]
    wall_count = 0

    for (delta_x, delta_y) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        if np.all(maze[current[1] + delta_y, current[0] + delta_x] == BLACK_PIXEL):
            wall_count += 1

    if wall_count == 2:
        for (delta_x, delta_y) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (np.all(maze[current[1] + delta_y, current[0] + delta_x] == WHITE_PIXEL) and visited[current[0] + delta_x, current[1] + delta_y] == 0):
                return (current[0] + delta_x, current[1] + delta_y), (current[0], current[1])

    else:
        for (delta_x, delta_y) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (np.all(maze[current[1] + delta_y, current[0] + delta_x] == WHITE_PIXEL) and visited[current[0] + delta_x, current[1] + delta_y] == 0):
                if current[1] + delta_y == height - 1:
                    return (current[0] + delta_x, current[1] + delta_y), (current[0], current[1])
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if (np.all(maze[current[1] + delta_y + dy, current[0] + delta_x + dx] == BLACK_PIXEL) and [current[1] + delta_y + dy, current[0] + delta_x + dx] != current):
                        return (current[0] + delta_x, current[1] + delta_y), (current[0], current[1])

    wall_side: Orientation

    for direction, (dx, dy) in {
        Orientation.BOTTOM: (0, 1),
        Orientation.TOP: (0, -1),
        Orientation.LEFT: (-1, 0),
        Orientation.RIGHT: (1, 0)
    }.items():
        if np.all(maze[current[1] + dy, current[0] + dx] == BLACK_PIXEL):
            wall_side = direction

    if wall_side in [Orientation.BOTTOM, Orientation.TOP]:
        for (dx, dy) in [(-1, 0), (1, 0)]:
            if (np.all(maze[current[1] + dy, current[0] + dx] == WHITE_PIXEL) and visited[current[0] + dx, current[1] + dy] == 0):
                return (current[0] + dx, current[1] + dy), (current[0], current[1])

    if wall_side in [Orientation.LEFT, Orientation.RIGHT]:
        for (dx, dy) in [(0, -1), (0, 1)]:
            if (np.all(maze[current[1] + dy, current[0] + dx] == WHITE_PIXEL) and visited[current[0] + dx, current[1] + dy] == 0):
                return (current[0] + dx, current[1] + dy), (current[0], current[1])


def find_way_from_maze(maze_image: np.ndarray) -> tuple[list[int], list[int]]:
    """
    Находит путь через лабиринт.

    :param maze_image: Изображение лабиринта.
    :return: Списки координат x и y пути.
    """
    height, width = maze_image.shape[:2]
    visited = np.zeros((height, width), dtype=int)

    start, _= locate_maze_openings(maze_image)
    visited[start[0], start[1]] = 1

    path = [start, (start[0], start[1] + 1)]
    visited[start[0], start[1] + 1] = 1

    current = (start[0], start[1] + 1)
    steps = 0

    while True:
        next_step, _ = determine_next_position(maze_image, current, visited)
        visited[next_step[0], next_step[1]] = 1
        path.append(next_step)
        if next_step[1] == height - 1:
            break
        current = next_step
        steps += 1

    return [x for x, _ in path], [y for _, y in path]