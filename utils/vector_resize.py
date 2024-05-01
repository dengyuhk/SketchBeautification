import numpy as np
import cv2 as cv


white = (256,256,256)

gray = (100,100,100)
colors = [
    (0, 0, 255),  # airplane
    (255, 0, 0),  # apple
    (0, 255, 0),  # balloon
    (255, 26, 185),  # banana
    (255, 211, 0),  # basket
    (0, 132, 246),  # bee
    (0, 141, 70),  # bench
    (0, 62, 53),  # bicycle
    (167, 97, 62),  # bird
    (0, 62, 53),  # bottle
    (0, 62, 53),  # bucket
    (79, 0, 106),  # bus
    (0, 255, 246),  # butterfly
    (62, 123, 141),  # car
    (237, 167, 255),  # cat
    (211, 255, 149),  # chair
    (185, 79, 255),  # chicken
    (229, 26, 88),  # cloud
    (132, 132, 0),  # cow
    (0, 62, 53),  # cup
    (0, 62, 53),  # dinnerware
    (0, 255, 149),  # dog
    (97, 0, 44),  # duck
    (246, 132, 18),  # fence
    (202, 255, 0),  # flower
    (44, 62, 0),  # grape
    (0, 53, 193),  # grass
    (255, 202, 132),  # horse
    (0, 44, 97),  # house
    (158, 114, 141),  # moon
    (79, 185, 18),  # mountain
    (158, 193, 255),  # people
    (149, 158, 123),  # picnic rug
    (255, 123, 176),  # pig
    (158, 9, 0),  # rabbit
    (255, 185, 185),  # road
    (132, 97, 202),  # sheep
    (0, 62, 53),  # sofa
    (158, 0, 114),  # star
    (132, 220, 167),  # street lamp
    (255, 0, 246),  # sun
    (0, 211, 255),  # table
    (255, 114, 88),  # tree
    (88, 62, 53),  # trunk
    (0, 62, 53),  # umbrella
    (0, 62, 53),  # others
]


segcolor = (100, 100, 100)
sscolor = (255, 0, 246)

def neighbours(x, y, image):
    '''Return 8-neighbours of point p1 of picture, in order'''
    i = image
    x1, y1, x_1, y_1 = x + 1, y - 1, x - 1, y + 1
    # print ((x,y))
    return [i[y1][x], i[y1][x1], i[y][x1], i[y_1][x1],  # P2,P3,P4,P5
            i[y_1][x], i[y_1][x_1], i[y][x_1], i[y1][x_1]]  # P6,P7,P8,P9


def transitions(neighbours):
    n = neighbours + neighbours[0:1]  # P2, ... P9, P2
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))


def zhangSuen(image):
    changing1 = changing2 = [(-1, -1)]
    while changing1 or changing2:
        # Step 1
        changing1 = []
        for y in range(1, len(image) - 1):
            for x in range(1, len(image[0]) - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, image)
                if (image[y][x] == 1 and  # (Condition 0)
                        P4 * P6 * P8 == 0 and  # Condition 4
                        P2 * P4 * P6 == 0 and  # Condition 3
                        transitions(n) == 1 and  # Condition 2
                        2 <= sum(n) <= 6):  # Condition 1
                    changing1.append((x, y))
        for x, y in changing1: image[y][x] = 0
        # Step 2
        changing2 = []
        for y in range(1, len(image) - 1):
            for x in range(1, len(image[0]) - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, image)
                if (image[y][x] == 1 and  # (Condition 0)
                        P2 * P6 * P8 == 0 and  # Condition 4
                        P2 * P4 * P8 == 0 and  # Condition 3
                        transitions(n) == 1 and  # Condition 2
                        2 <= sum(n) <= 6):  # Condition 1
                    changing2.append((x, y))
        for x, y in changing2: image[y][x] = 0
        # print changing1
        # print changing2
    return image


def generate_single(data):
    mat = (data > 0).tolist()
    after = zhangSuen(mat)
    after = np.array(after)

    new_data = data * after

    return new_data

def sample_point2(stroke_set, label_list):
    point_sequence = []
    label_sequence = []
    order_sequence = []
    # point_counter, stroke_counter, rate = count(stroke_set, K)
    for index, data in enumerate(zip(stroke_set, label_list)):
        stroke, label = data
        # num = int(len(stroke) * rate)
        # print('{}:{}'.format(len(stroke), num))
        # if True:
        if len(stroke) >= 2:
            point_sequence += stroke
            label_sequence += [label] * len(stroke)
            order_sequence += [index] * len(stroke)

            # selected = np.linspace(0, len(stroke)-1, num=num, dtype=np.int16)
            # point_sequence += [stroke[i] for i in selected]
            # label_sequence += [label] * num
            # order_sequence += [index] * num
    # print(':' + str(len(point_sequence)))
    # flag = False
    # if len(point_sequence) < 250:
    #     flag = True
    return point_sequence, label_sequence, order_sequence

def get_final_stroke(points, labels, order):
    final_stroke = []
    final_label = []
    stroke = []
    label = []
    part_id = order[0]
    for index, point in enumerate(points):
        if part_id != order[index]:
            final_stroke.append(stroke)
            final_label.append(label)
            part_id = order[index]
            stroke = []
            label = []
            stroke.append(point)
            label.append(labels[index])
        else:
            stroke.append(point)
            label.append(labels[index])

    final_stroke.append(stroke)
    final_label.append(label)
    return final_stroke, final_label

def trans(final_stroke, final_label):
    final = []
    for stroke, label in zip(final_stroke, final_label):
        s = np.vstack((np.array(stroke).T, np.array(label))).tolist()
        final.append(s)
    return final


def drawColor(sketch, imgSizes, zoomTimes=1,hideFlag=False):
    # penColor = white
    canvas = np.ones((imgSizes,imgSizes,3),dtype='uint8')*255
    # counter = 0

    # if hideFlag:
    #     penColor = white
    #     for stroke in sketch:
    #         for i in range(1, len(stroke[0])):
    #             cv.line(canvas,
    #                     (int(stroke[0][i-1]*zoomTimes), int(stroke[1][i-1]*zoomTimes)),
    #                     (int(stroke[0][i]*zoomTimes), int(stroke[1][i]*zoomTimes)),
    #                     penColor)
    # else:
    for stroke in sketch:
        # penColor = colors[int(stroke[2][0])+1]
        # cv.circle(canvas,
        #           (int(stroke[0][0]*zoomTimes), int(stroke[1][0]*zoomTimes)),
        #           3, penColor)
        for i in range(1, len(stroke[0])):
            penColor = colors[int(stroke[2][i])]
            cv.line(canvas,
                    (int(stroke[0][i-1]*zoomTimes), int(stroke[1][i-1]*zoomTimes)),
                    (int(stroke[0][i]*zoomTimes), int(stroke[1][i]*zoomTimes)),
                    penColor)
            # cv.circle(canvas,
            #           (int(stroke[0][i]*zoomTimes), int(stroke[1][i]*zoomTimes)),
            #           2, sscolor)
        # cv.circle(canvas,
        #           (int(stroke[0][-1]*zoomTimes), int(stroke[1][-1]*zoomTimes)),
        #           3, sscolor)

    return np.array(canvas)