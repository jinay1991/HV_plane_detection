import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
import random
import os



class segmentation():

    def __init__(self, filename, isDebugWindowsVisible=False, minContourArea=200):
        self.isDebugWindowsVisible = isDebugWindowsVisible
        self.filename = filename
        self.contours = []
        self.MIN_CONTOUR_AREA_REQUIRED = minContourArea

    def preprocess(self, filename):
        self.img = cv2.imread(filename)
        if self.img is None:
            print "file does not exist"
            return

        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # noise removal
        blur = cv2.medianBlur(self.img, 3)
        blur = cv2.GaussianBlur(blur, (3, 3), 0)
        blur = cv2.blur(blur, (3, 3))

        # convert to GRAYSCALE
        self.G = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)


        # Noise removal
        G_filtered = cv2.bilateralFilter(self.G, 9, 30, 30)

        # threshold
        _, G_thresh = cv2.threshold(G_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return G_thresh

    def get_contours(self, image):
        # Find contours for the Morphed Image
        ret, t_contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        t_contours = [cv2.approxPolyDP(c, epsilon=2.5, closed=True) for c in t_contours]
        contour_plot = np.zeros(image.shape[:2], dtype=np.uint8)
        for c in t_contours:
            if cv2.contourArea(c) < self.MIN_CONTOUR_AREA_REQUIRED:
                continue
            self.contours.append(c)

        return self.contours

    def floor_segmentation(self):

        # pre-process the input
        G_thresh = self.preprocess(self.filename)

        filteredContours = self.get_contours(G_thresh)

        contour_plot = np.zeros(G_thresh.shape[:2], dtype=np.uint8)
        cv2.drawContours(contour_plot, filteredContours, -1, 255, thickness=-1)

        # dilate and edge detection on the detected contour plot in order to fill the gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        G_morph = cv2.dilate(contour_plot, kernel, iterations=1)
        G_edges = cv2.Canny(G_morph, 40, 100, apertureSize=3)

        if self.isDebugWindowsVisible:
            plt.subplot(221), plt.title("G"), plt.imshow(self.G, cmap='gray')
            plt.subplot(222), plt.title("G_thresh"), plt.imshow(G_thresh, cmap='gray')
            plt.subplot(223), plt.title("contour_plot"), plt.imshow(contour_plot, cmap='gray')
            plt.subplot(224), plt.title("G_edges (After Contours)"), plt.imshow(G_edges, cmap='gray')
            plt.show()

        self.floorEdgePoints = []
        height = G_edges.shape[0] - 1
        width = G_edges.shape[1] - 1
        for j in range(0, width, 8):
            for i in range(height - 20, 0, -1):
                if G_edges.item(i,j) == 255:
                    self.floorEdgePoints.append((j,i))
                    break
            else:
                self.floorEdgePoints.append((j,0))

        self.dump_output(self.floorEdgePoints, width, height)

        return self.floorEdgePoints


    def dump_output(self, pts, width, height):

        clean_img = self.img.copy()
        mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        for x in range(len(pts)-1):
            cv2.line(clean_img, pts[x], pts[x+1], (180,255,0), 2)
            cv2.line(mask, pts[x], pts[x+1], (255, 0, 0), 2)

        mask = cv2.bitwise_not(mask)

        for i in range(0,width):
            for j in range(0,height):
                if mask[j,i] != 255:
                    break
                else:
                    mask[j,i] = 0
        B = clean_img[:,:,2]
        B[mask > 0] = 255

        dumpFileName = str(os.path.splitext(os.path.basename(self.filename))[0]) + "_OUT.jpg"
        if self.isDebugWindowsVisible:
            plt.imshow(clean_img), plt.title(dumpFileName), plt.show()

        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(dumpFileName, clean_img)
        del clean_img


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input file path", default="data/IMG_6857.JPG")
    parser.add_argument("--debug", help="display intermediate outputs", default=1)
    args = parser.parse_args()

    segment = segmentation(filename=args.input, isDebugWindowsVisible=args.debug)
    floor_boundary_pts = segment.floor_segmentation()