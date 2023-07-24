import os

import cv2
import gdown
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch.nn as nn

from model.MEFL import MEFARG
from OpenGraphAU.model.face_detection import SCRFD_ONNX, ExpandBbox
from utils import *

# Set the style and context of the plot
sns.set(rc={"figure.figsize": (25, 5)})
sns.set_style("whitegrid")
sns.set_context("notebook", rc={"lines.linewidth": 3})


class MEGraphAU:
    def __init__(
        self,
        resume="/home/os/Downloads/ME-GraphAU_resnet50_DISFA/MEFARG_resnet50_DISFA_fold2.pth",
        num_classes=8,
        arc="resnet50",
        device="cpu",
    ):
        self.device = torch.device(device)
        self.expand = ExpandBbox()
        self.transform = image_test()
        self.face_detector = SCRFD_ONNX("checkpoints/face_detection/scrfd_500.onnx")
        self.net = MEFARG(
            num_classes=num_classes,
            backbone=arc,
        )

        # resume
        if resume != "":
            self.net = load_state_dict(self.net, resume)

        self.net.eval()
        if device == "cuda":
            self.net == nn.DataParallel(self.net).to(self.device)

    def face_detection(self, image, expand=True):
        bbox = self.face_detector.run(image)
        if expand:
            bbox = self.expand(image, bbox)
        return bbox

    def _process_image(self, image):
        image = cv2.imread(image) if type(image) == "str" else image

        bbox = self.face_detection(image)
        image_cropped = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        image_cropped = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB)
        image_cropped = Image.fromarray(image_cropped)
        image_cropped = self.transform(image_cropped).unsqueeze(0).to(self.device)
        return image_cropped

    def run(self, image):
        image_cropped = self._process_image(image)
        with torch.no_grad():
            pred, _ = self.net(image_cropped)
            pred = pred.squeeze().cpu().numpy()
        pred = DISFA_infolist(pred)
        print(pred)
        return pred


def main():
    me_graph_au = MEGraphAU()
    cap = cv2.VideoCapture(1)
    cap.set(3, 1280)
    cap.set(4, 720)
    # cap = cv2.VideoCapture("girl_expression.mp4")

    while True:
        _, frame = cap.read()
        if not _:
            break
        frame = cv2.flip(frame, 1)
        result = me_graph_au.run(frame)

        # data = pd.DataFrame.from_dict(result[1], orient="index", columns=["Intensity"])
        #
        # plt.figure().clf()  # Clear the figure after every frame
        # sns.barplot(x=data.index, y="Intensity", data=data).figure.savefig("au_intensity.png")
        # cv2.imshow("AU Intensity", cv2.imread("au_intensity.png"))
        cv2.imshow("WEBCAM", frame)
        if cv2.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":
    main()
