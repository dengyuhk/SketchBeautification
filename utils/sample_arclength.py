import numpy as np


def lerp(v0, v1, t):
    return (1 - t) * v0 + t * v1

def sample(points,s):
    """
    :param points: points[P0,P1,Pn-1] raw points
    :param s: arclength
    :return: point on P ar arclength s
    """
    cur_length = 0
    for i in range(len(points) - 1):
        length = np.linalg.norm(points[i + 1] - points[i])
        if cur_length + length >= s:
            alpha = (s - cur_length) / length
            result = lerp(points[i], points[i + 1], alpha)
            return result
        cur_length += length

def resample(points, inc=0.01):
    """
    Given:
        points: [P0, P1, ..., Pn-1] raw points
        inc: sampling rate of the curve, 1cm
             can modify to other sampling rate, e.g. how many points
    """
    length = 0
    for i in range(len(points) - 1):
        length += np.linalg.norm(points[i + 1] - points[i])

    num = int(length / inc)

    q = []
    for i in range(num):
        q.append(sample(points, i * inc))

    q.append(points[-1,:])


    return q

def resample_uniform(points, num=0):
    """
    Given:
        points: [P0, P1, ..., Pn-1] raw points
        inc: sampling rate of the curve, 1cm
             can modify to other sampling rate, e.g. how many points
    """
    length = 0
    for i in range(len(points) - 1):
        length += np.linalg.norm(points[i + 1] - points[i])

    inc = int(length / num)

    q = []
    for i in range(num):
        q.append(sample(points, i * inc))


    return q