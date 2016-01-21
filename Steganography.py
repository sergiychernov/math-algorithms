import cv2
import numpy

img = cv2.imread('images/lena512.bmp')
B, G, R = cv2.split(img)

X = len(B)
Y = len(B[0])
N = 8
Nc = X * Y / (N ^ 2)
v1 = 2
v2 = 3
u1 = 2
u2 = 3
P1 = 50
P2 = 10
c1 = 0
c2 = N
s = 'here will be some long long text or even longer than you imagine'
M = map(int, ''.join([bin(ord(i)).lstrip('0b').rjust(8, '0') for i in s]))

Lm = len(M)
print Lm
Tmp = [Nc]

for b in range(0, Lm):
    r1 = (N * (b - 1)) % X
    r2 = r1 + N
    if c1 <= Y and c2 <= Y and r1 <= X and r2 <= X and c1 >= 0 and c2 > 0 and r1 > 0 and r2 > 0:
        Cb = B[r1:r2, c1:c2]

        U, S, V = numpy.linalg.svd(Cb)
        w1 = V[u1, v1]
        w2 = V[u2, v2]

        if M[b] == 0:
            if w1 - w2 != P1:
                w1 = P1 + w2

        if M[b] == 1:
            if w1 - w2 != P2:
                w1 = P2 + w2
        V[u1, v1] = w1
        V[u2, v2] = w2
        print b
        B[r1:r2, c1:c2] = numpy.dot(U, numpy.dot(S, V.transpose()))
        if r2 == X:
            c1 += N
            c2 += N

img1 = cv2.merge((R, G, B))
cv2.imwrite('images/lena512_processed.bmp', img1)
print str(X) + 'x' + str(Y)
