# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 All Rights Resevered
#
# Author: Jinay Patel (jinay1991@gmail.com)

"""
Horizontal and Vertical Surface detection (i.e. Floor/Wall Segmentation)

usage: HV_plane_detect.py [-h] --input INPUT [--debug]

optional arguments:
  -h, --help     show this help message and exit
  --input INPUT  input file path
  --debug        display intermediate outputs
"""
import cv2
import numpy as np
import os
import logging

log = logging.getLogger()
logging.basicConfig(level=logging.INFO)


class segmentation():

    def __init__(self, filename):
        self.filename = filename
        self.__preprocess(self.filename)

    def __preprocess(self, filename):
        """
        Loads Image and performs basic Preprocessing. [Private Method]
        Preprocessing includes (BGR->RGB, filters, RGB->Gray, Gray->Thresh)
        """
        self.in_image = cv2.imread(filename)

        self.rgb_image = cv2.cvtColor(self.in_image, cv2.COLOR_BGR2RGB)

        # noise removal
        # blur = cv2.medianBlur(self.rgb_image, 3)
        # blur = cv2.GaussianBlur(blur, (3, 3), 0)
        # blur = cv2.blur(blur, (3, 3))
        # noise removal
        blur = cv2.bilateralFilter(self.rgb_image, 15, 90, 90)

        # convert to GRAYSCALE
        self.G = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)

        # Noise removal

        # threshold
        _, self.G_thresh = cv2.threshold(
            self.G, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def __compute_contours(self, image, minContourArea=100):
        """
        Computes Contours for the provided Image. [Private Method]
        """
        # Find contours for the Morphed Image
        ret, t_contours, hierarchy = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        t_contours = [cv2.approxPolyDP(
            c, epsilon=2.5, closed=True) for c in t_contours]

        contours = []
        for c in t_contours:
            if cv2.contourArea(c) < minContourArea:
                continue
            contours.append(c)

        return contours

    def wall_segmentation(self):
        """
        Segmentation for the Wall Extraction from the image

        Returns: Wall Contour Points, Wall Mask
        """
        wall_img = self.rgb_image.copy()
        G_thresh = self.G_thresh.copy()

        if self.floor_mask is None:
            _, self.floor_mask = self.floor_segmentation()

        # remove portion of Floor (already detected pixels)
        G_thresh[self.floor_mask == 255] = 0
        # remove sky (usually tends to have higher gray values)
        # sky_mask = cv2.inRange(self.rgb_image, np.array([90,127,127]), np.array([90,127,127]))
        # G_thresh[sky_mask > 0 ] = 0

        # perform watershed algorithm to segment walls from the background
        # _, markers = cv2.connectedComponents(G_thresh)
        # markers = markers + 1
        # markers = cv2.watershed(wall_img, markers)

        self.wall_pts = self.__compute_contours(G_thresh)

        self.wall_mask = np.zeros(G_thresh.shape[:2], dtype=np.uint8)
        cv2.drawContours(self.wall_mask, self.wall_pts, -1, 255, thickness=-1)

        # Give highlighter to image
        cv2.drawContours(wall_img, self.wall_pts, -1, (255, 0, 0), thickness=2)
        G = wall_img[:, :, 1]
        G[self.wall_mask > 0] = 255

        if log.level == logging.DEBUG:
            # plt.subplot(131), plt.imshow(markers), plt.title("markers")
            plt.subplot(121), plt.imshow(
                G_thresh, cmap='gray'), plt.title("G_thresh")
            plt.subplot(122), plt.imshow(wall_img), plt.title("Walls")
            plt.show()

        return self.wall_pts, self.wall_mask

    def color_quantisation(self):
        
        img = cv2.imread(self.filename)
        Z = img.reshape((-1,3))
        
        # convert to np.float32
        Z = np.float32(Z)
        
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 8
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        
        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        
        plt.subplot(121), plt.imshow(res2), plt.title("res2") 
        gray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        plt.subplot(122), plt.imshow(thresh, cmap='gray'), plt.title("thresh")
        plt.show()
        
    def houghLines(self, probabilistic=False):
        """
        Computes Hough Transforms
        """
        HoughLines_img = self.rgb_image.copy()
        filteredContours = self.__compute_contours(
            self.G_thresh, minContourArea=200)

        contour_plot = np.zeros(self.G_thresh.shape[:2], dtype=np.uint8)
        cv2.drawContours(contour_plot, filteredContours, -1, 255, thickness=-1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
        morph = cv2.dilate(contour_plot, kernel, iterations=1)
        edges = cv2.Canny(morph, 20, 40, apertureSize=3)

        self.houghline_G_thresh_mask = np.zeros(edges.shape, dtype=np.uint8)

        if not probabilistic:
            draw_H = np.zeros(HoughLines_img.shape, dtype='uint8')
            draw_V = np.zeros(HoughLines_img.shape, dtype='uint8')
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 10)
            if lines is not None:
                for line in lines[:10]:
                    i = 0
                    for rho, theta in line:
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))
                        theta_in_degree = theta * 180 / np.pi
                        i += 1
                        if theta_in_degree > 180 - 30 or theta_in_degree < 30:
                            line_color = (0, 255, 0)
                            draw_on = draw_V
                        elif theta_in_degree > 90 - 45 and theta_in_degree < 90 + 45:
                            line_color = (0, 0, 255)
                            draw_on = draw_H
                            cv2.line(self.houghline_G_thresh_mask,
                                     (x1, y1), (x2, y2), 255, 2)
                        else:
                            continue

                        cv2.line(draw_on, (x1, y1), (x2, y2), line_color, 2)
                        cv2.line(HoughLines_img, (x1, y1),
                                 (x2, y2), line_color, 2)
            else:
                print "no lines found"
            if log.level == logging.DEBUG:
                plt.subplot(221), plt.imshow(HoughLines_img), plt.title("img")
                plt.subplot(222), plt.imshow(draw_H), plt.title("line_H")
                plt.subplot(223), plt.imshow(draw_V), plt.title("line_V")
                plt.subplot(224), plt.imshow(
                    draw_H + draw_V), plt.title("line_H + line_V")
                plt.show()
        else:
            lines_V = []
            lines_H = []
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 2, 10, None, minLineLength=15, maxLineGap=10)
            if lines is not None:
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        angle = round(np.arctan2(y2 - y1, x2 - x1)
                                      * 180. / np.pi, 2)
                        # print "angle:", angle
                        if angle < 45 and angle > -45:  # Horizontal Lines
                            lines_H.append(((x1, y1), (x2, y2)))
                        elif (angle > 85 and angle < 95) or (angle > -95 and angle < -85):  # Vertical Lines
                            lines_V.append(((x1, y1), (x2, y2)))
                        else:
                            continue
            else:
                print "no lines found with HoughLineP"

            img_lineV = np.zeros(HoughLines_img.shape, dtype=np.uint8)
            for ((x1, y1), (x2, y2)) in lines_V:
                cv2.line(img_lineV, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.line(HoughLines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            img_lineH = np.zeros(HoughLines_img.shape, dtype=np.uint8)
            for ((x1, y1), (x2, y2)) in lines_H:
                cv2.line(img_lineH, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.line(HoughLines_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.line(self.houghline_G_thresh_mask,
                         (x1, y1), (x2, y2), 255, 2)

            if log.level == logging.DEBUG:
                plt.subplot(221), plt.imshow(self.houghline_G_thresh_mask), plt.title(
                    "houghline_G_thresh_mask")
                plt.subplot(222), plt.imshow(
                    HoughLines_img), plt.title("HoughLineP")
                plt.subplot(223), plt.imshow(img_lineH), plt.title("img_lineH")
                plt.subplot(224), plt.imshow(img_lineV), plt.title("img_lineV")
                plt.show()

        return self.houghline_G_thresh_mask

    def floor_segmentation(self):

        # pre-process the input
        G_edges = self.houghLines(False)  # G_thresh.copy()

        floorEdgePoints = []
        height = G_edges.shape[0] - 1
        width = G_edges.shape[1] - 1
        for j in range(0, width, 8):
            for i in range(height - 10, 0, -1):
                if G_edges.item(i, j) == 255:
                    floorEdgePoints.append((j, i))
                    break
            else:
                floorEdgePoints.append((j, height - 20))

        for (x, y) in floorEdgePoints:
            if y < height / 2:
                floorEdgePoints.remove((x,y))

        import operator
        floorEdgePoints.sort(key=operator.itemgetter(0), reverse=True)

        mask = np.zeros(G_edges.shape[:2], dtype=np.uint8)
        for x in range(len(floorEdgePoints) - 1):
            cv2.line(mask, floorEdgePoints[x], floorEdgePoints[
                     x + 1], 255, 1)

        self.floor_pts = self.__compute_contours(mask, minContourArea=0)
        self.floor_mask = cv2.bitwise_not(mask)

        for i in range(0, width):
            for j in range(0, height):
                if self.floor_mask[j, i] != 255:
                    break
                else:
                    self.floor_mask[j, i] = 0

        # Give highlighter to image
        floor_img = self.rgb_image.copy()
        cv2.drawContours(floor_img, self.floor_pts, -
                         1, (0, 255, 0), thickness=2)
        B = floor_img[:, :, 2]
        B[self.floor_mask > 0] = 255

        if log.level == logging.DEBUG:
            plt.subplot(131), plt.title("G"), plt.imshow(self.G, cmap='gray')
            plt.subplot(132), plt.title(
                "G_edges"), plt.imshow(G_edges, cmap='gray')
            plt.subplot(133), plt.title("floor_img"), plt.imshow(floor_img)
            plt.show()

        return self.floor_pts, self.floor_mask

    def dump_results(self):

        clean_img = self.rgb_image.copy()
        # mask = np.zeros(self.img.shape[:2], dtype=np.uint8)

        cv2.drawContours(clean_img, self.floor_pts, -1, (180, 250, 0), 2)
        cv2.drawContours(clean_img, self.wall_pts, -1, (255, 180, 0), 2)

        R, G, B = clean_img[:, :, 0], clean_img[:, :, 1], clean_img[:, :, 2]

        B[self.floor_mask > 0] = 255
        R[self.wall_mask > 0] = 255

        dumpFileName = str(os.path.splitext(
            os.path.basename(self.filename))[0]) + "_OUT.jpg"
        if log.level == logging.DEBUG:
            plt.imshow(clean_img), plt.title(dumpFileName), plt.show()

        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(dumpFileName, clean_img)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input file path",
                        default="data/IMG_6857.JPG", required=True)
    parser.add_argument("--debug", help="display intermediate outputs, requires matplotlib.pyplot",
                        action="store_true", default=False)
    args = parser.parse_args()

    if args.debug:
        from matplotlib import pyplot as plt
        log.setLevel(logging.DEBUG)

    if not os.path.exists(os.path.abspath(args.input)):
        log.error("input file does not exist")
        exit(1)

    segment = segmentation(filename=args.input)

    floor_pts, floor_mask = segment.floor_segmentation()
    wall_pts, wall_mask = segment.wall_segmentation()

    segment.dump_results()
