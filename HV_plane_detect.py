import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
import random

display_DEBUG = False

def main(filename):
    img = cv2.imread(filename)
    if img is None:
        print "file does not exist"
        return
    
    clean_img = img.copy()
    # noise removal
    blur = cv2.medianBlur(img, 3)
    blur = cv2.GaussianBlur(blur, (3, 3), 0)
    blur = cv2.blur(blur, (3, 3))

    hsv_image = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    H, S, V = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]
    # V = V * 2
    # hsv_image = cv2.merge([H, S, V])
    G = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # display HSV Channels
    if display_DEBUG:
        plt.subplot(221), plt.imshow(H, cmap='gray'), plt.title("H")
        plt.subplot(222), plt.imshow(S, cmap='gray'), plt.title("S")
        plt.subplot(223), plt.imshow(V, cmap='gray'), plt.title("V")
        plt.subplot(224), plt.imshow(G, cmap='gray'), plt.title("G")
        plt.show()

    _, H_thresh = cv2.threshold(H, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, S_thresh = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, V_thresh = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, G_thresh = cv2.threshold(G, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # display HSV Channels
    if display_DEBUG:
        plt.subplot(221), plt.imshow(H_thresh, cmap='gray'), plt.title("H_thresh")
        plt.subplot(222), plt.imshow(S_thresh, cmap='gray'), plt.title("S_thresh")
        plt.subplot(223), plt.imshow(V_thresh, cmap='gray'), plt.title("V_thresh")
        plt.subplot(224), plt.imshow(G_thresh, cmap='gray'), plt.title("G_thresh")
        plt.show()

    H_edges = cv2.Canny(H, 50, 100, apertureSize=3)
    S_edges = cv2.Canny(S, 50, 100, apertureSize=3)
    V_edges = cv2.Canny(V, 50, 100, apertureSize=3)
    G_edges = cv2.Canny(G, 50, 100, apertureSize=3)

    # # display HSV Channels
    if display_DEBUG:
        plt.subplot(221), plt.imshow(H_edges, cmap='gray'), plt.title("H_edges")
        plt.subplot(222), plt.imshow(S_edges, cmap='gray'), plt.title("S_edges")
        plt.subplot(223), plt.imshow(V_edges, cmap='gray'), plt.title("V_edges")
        plt.subplot(224), plt.imshow(V_edges, cmap='gray'), plt.title("G_edges")
        plt.show()

    c = img.copy()
    lines = cv2.HoughLines(G_edges,1,np.pi/180,int(img.shape[1] * 0.20))
    draw_H = np.zeros(img.shape,dtype='uint8')
    draw_V = np.zeros(img.shape,dtype='uint8')
    if lines is not None:
        for line in lines[:10]:
            i = 0
            for rho,theta in line:
                theta_in_degree = theta * 180/np.pi
                print "theta for line[", i, "] is", theta_in_degree
                i += 1
                if theta_in_degree > 180 - 65 or theta_in_degree < 65:
                    line_color = (0, 255, 0)
                    draw_on = draw_V
                elif theta_in_degree > 90 - 10 and theta_in_degree < 90 + 10:
                    line_color = (0, 0, 255)
                    draw_on = draw_H
                else:
                    print "continue"
                    continue
                print "drawing line on"
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(draw_on,(x1,y1),(x2,y2),line_color,2)
                cv2.line(img,(x1,y1),(x2,y2),line_color,2)
    else:
        print "no lines found"

    if display_DEBUG:
        plt.subplot(221), plt.imshow(clean_img), plt.title("clean_img")
        plt.subplot(222), plt.imshow(draw_H), plt.title("line_H")
        plt.subplot(223), plt.imshow(draw_V), plt.title("line_V")
        plt.subplot(224), plt.imshow( draw_H + draw_V), plt.title("line_H + line_V")
        plt.show()
        plt.imshow(img), plt.title("img")
        plt.show()
        

    lines = cv2.HoughLinesP(G_edges,1, np.pi/2, 2, None, 20, 1)
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
                if angle > 180 - 65 or angle < 65:
                    line_color = (0, 255, 0)
                elif angle > 90 - 10 and angle < 90 + 10:
                    line_color = (0, 0, 255)
                else:
                    print "continue"
                    continue
                cv2.line(c,(x1,y1),(x2,y2),line_color,2)
    else:
        print "no lines found with HoughLineP"

    if display_DEBUG:
        plt.subplot(121), plt.imshow(clean_img), plt.title("clean_img")
        plt.subplot(122), plt.imshow(c), plt.title("HoughLineP")
        plt.show()

    cv2.imwrite("HoughLineP_OUT.jpg", c)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input file path")
    parser.add_argument("debug", help="display intermediate outputs")
    args = parser.parse_args()
    if args.debug != 0:
        display_DEBUG = True
    main(args.input)