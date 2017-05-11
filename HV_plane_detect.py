import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
import random
import os

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
    # V = 0
    # hsv_image = cv2.merge([H, S, V])
    # lower = np.array([40,60,60]) #np arrays for upper and lower thresholds
    # upper = np.array([100,200,230])

    # imgthreshed = cv2.inRange(hsv_image, lower, upper) #threshold imgHSV

    # plt.imshow(imgthreshed, cmap='gray'), plt.show()
    # exit(1)

    G = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    G = cv2.bilateralFilter(G, 9, 30, 30)


    # blended = (H + S + V) / 2
    _, G_thresh = cv2.threshold(G, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    G_close = cv2.dilate(G_thresh, kernel, iterations=1)

    G_edges = cv2.Canny(G_close, 50, 100, apertureSize=3)

    # # display HSV Channels
    if display_DEBUG:
        plt.subplot(221), plt.imshow(G, cmap='gray'), plt.title("G")
        plt.subplot(222), plt.imshow(G_thresh, cmap='gray'), plt.title("G_thresh")
        plt.subplot(223), plt.imshow(G_edges, cmap='gray'), plt.title("G_edges")
        plt.show()

    ret, contours, hierarchy = cv2.findContours(G_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    approx_contours = [cv2.approxPolyDP(c, epsilon=3, closed=True) for c in contours]
    contours = approx_contours
    filteredContours = []
    cnt_img = np.zeros(img.shape[:2], dtype=np.uint8)
    for c in contours:
        if cv2.contourArea(c) < 100:
            continue
        filteredContours.append(c)

    cv2.drawContours(cnt_img, filteredContours, -1, 255, thickness=-1)
    plt.imshow(cnt_img, cmap='gray'), plt.show()

    G_close = cv2.dilate(cnt_img, kernel, iterations=3)
    G_edges = cv2.Canny(G_close, 20, 40, apertureSize=3)

    plt.imshow(G_edges, cmap='gray'), plt.show()
    width = G_edges.shape[1] - 1
    height = G_edges.shape[0] - 1
    edgePoints = []
    for j in range(0, width, 8):
        for i in range(height - 20, 0, -1):
            if G_edges.item(i,j) == 255:
                edgePoints.append((j,i))
                break
        else:
            edgePoints.append((j,0))

    for x in range(len(edgePoints)-1):
        cv2.line(img, edgePoints[x], edgePoints[x+1], (180,255,0), 3)
    for x in range(len(edgePoints)):
        cv2.line(img, (x*8, height), edgePoints[x], (170,180,133),1)

    plt.subplot(121), plt.imshow(img)
    plt.show()

    dumpFileName = str(os.path.splitext(os.path.basename(filename))[0]) + "_OUT.jpg"
    cv2.imwrite(dumpFileName, img)
    exit(0);
    # lines = cv2.HoughLines(G_edges,1,np.pi/180,int(img.shape[1] * 0.20))
    # draw_H = np.zeros(img.shape,dtype='uint8')
    # draw_V = np.zeros(img.shape,dtype='uint8')
    # if lines is not None:
    #     for line in lines[:10]:
    #         i = 0
    #         for rho,theta in line:
    #             theta_in_degree = theta * 180/np.pi
    #             print "theta for line[", i, "] is", theta_in_degree
    #             i += 1
    #             if theta_in_degree > 180 - 65 or theta_in_degree < 65:
    #                 line_color = (0, 255, 0)
    #                 draw_on = draw_V
    #             elif theta_in_degree > 90 - 10 and theta_in_degree < 90 + 10:
    #                 line_color = (0, 0, 255)
    #                 draw_on = draw_H
    #             else:
    #                 print "continue"
    #                 continue
    #             print "drawing line on"
    #             a = np.cos(theta)
    #             b = np.sin(theta)
    #             x0 = a*rho
    #             y0 = b*rho
    #             x1 = int(x0 + 1000*(-b))
    #             y1 = int(y0 + 1000*(a))
    #             x2 = int(x0 - 1000*(-b))
    #             y2 = int(y0 - 1000*(a))
    #             cv2.line(draw_on,(x1,y1),(x2,y2),line_color,2)
    #             cv2.line(img,(x1,y1),(x2,y2),line_color,2)
    # else:
    #     print "no lines found"
    # if display_DEBUG:
    #     plt.subplot(221), plt.imshow(img), plt.title("img")
    #     plt.subplot(222), plt.imshow(draw_H), plt.title("line_H")
    #     plt.subplot(223), plt.imshow(draw_V), plt.title("line_V")
    #     plt.subplot(224), plt.imshow( draw_H + draw_V), plt.title("line_H + line_V")
    #     plt.show()

    lines_V = []
    lines_H = []
    lines = cv2.HoughLinesP(G_close,1, np.pi/2, 2, None, minLineLength=10, maxLineGap=2)
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                angle = round(np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi, 2)
                # print "angle:", angle
                if angle < 45 and angle > -45: # Horizontal Lines
                    lines_H.append(((x1,y1),(x2,y2)))
                elif (angle > 85 and angle < 95) or (angle > -95 and angle < -85) : # Vertical Lines
                    lines_V.append(((x1,y1),(x2,y2)))
                else:
                    continue
    else:
        print "no lines found with HoughLineP"

    HoughLines_img = img.copy()
    img_lineV = np.zeros(img.shape, dtype=np.uint8)
    for ((x1,y1),(x2,y2)) in lines_V:
        cv2.line(img_lineV,(x1,y1),(x2,y2),(0, 255, 0),2)
        cv2.line(HoughLines_img,(x1,y1),(x2,y2),(0,255,0),2)

    img_lineH = np.zeros(img.shape, dtype=np.uint8)
    for ((x1,y1),(x2,y2)) in lines_H:
        cv2.line(img_lineH,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.line(HoughLines_img,(x1,y1),(x2,y2),(0,0,255),2)

    if display_DEBUG:
        plt.subplot(221), plt.imshow(clean_img), plt.title("clean_img")
        plt.subplot(222), plt.imshow(HoughLines_img), plt.title("HoughLineP")
        plt.subplot(223), plt.imshow(img_lineH), plt.title("img_lineH")
        plt.subplot(224), plt.imshow(img_lineV), plt.title("img_lineV")
        plt.show()

    output = G.copy()
    maskH = img_lineH[:,:,2]
    maskV = img_lineV[:,:,1]
    output[maskH != 0] = 0
    output[maskV != 0] = 0
    plt.imshow(output, cmap='gray'), plt.show()
    cv2.imwrite("HoughLineP_OUT.jpg", output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input file path", default="data/IMG_6857.JPG")
    parser.add_argument("--debug", help="display intermediate outputs", default=1)
    args = parser.parse_args()
    if args.debug != 0:
        display_DEBUG = True
    main(args.input)