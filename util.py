import numpy as np

dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]

def is_black_pixel(pixel):
    return pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0


def fill(stack, img, visited, set_black=False):
    elements = set()

    while len(stack) > 0:
        i, j = stack.pop()
        elements.add((i, j))

        if set_black is False:
            visited[(i, j)] = True
        else:
            img[i, j] = 0

        for k in range(4):
            new_i = i + dx[k]
            new_j = j + dy[k]
            if new_i < 0 or new_i >= img.shape[0] or new_j < 0 or new_j >= img.shape[1]:
                continue
            if set_black is False:
                if visited.get((new_i, new_j), False) is True:
                    continue
            if is_black_pixel(img[new_i, new_j]):
                continue

            stack.append((new_i, new_j))
    return elements


def dilation(img, dim):
    filter = np.ones((dim, dim))
    img_h, img_w, _ = img.shape

    new_img = np.zeros()

    for i in range(img_h):
        for j in range(img_w):

            for filter_i in range(i - int(dim / 2), i + int(dim / 2) + 1):
                if filter_i < 0 or filter_i >= img_h:
                    continue
                for filter_j in range(j - int(dim / 2), j + int(dim / 2) + 1):
                    if filter_j < 0 or filter_j >= img_w:
                        continue
