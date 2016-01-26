import cv2
import numpy as np


def hide_text_in_image(b, text):
    x = len(b)
    y = len(b[0])
    n = 8
    nc = x * y / (n ^ 2)
    v1 = 2
    v2 = 3
    u1 = 2
    u2 = 3
    p1 = 50
    p2 = 10
    c1 = 0
    c2 = n
    msg = map(int, ''.join([bin(ord(i)).lstrip('0b').rjust(8, '0') for i in text]))

    for index in range(0, min(len(msg), nc)):
        r1 = (n * (index - 1)) % x
        r2 = r1 + n
        if c1 <= y and c2 <= y and r1 <= x and r2 <= x and c1 >= 0 and c2 > 0 and r1 > 0 and r2 > 0:
            cb = b[r1:r2, c1:c2]

            u, s, v = np.linalg.svd(cb)
            w1 = v[u1, v1]
            w2 = v[u2, v2]

            if msg[index] == 0:
                if w1 - w2 != p1:
                    w1 = p1 + w2

            if msg[index] == 1:
                if w1 - w2 != p2:
                    w1 = p2 + w2

            v[u1, v1] = w1
            v[u2, v2] = w2

            b[r1:r2, c1:c2] = np.dot(u, np.dot(np.diag(s), v))
            if r2 == x:
                c1 += n
                c2 += n


img = cv2.imread('images/lena512color.tiff')
B, g, r = cv2.split(img)

hide_text_in_image(B, 'here will be some long long text or even longer than you imagine')

cv2.imwrite('images/lena512_processed.tiff', cv2.merge((B, g, r)))
