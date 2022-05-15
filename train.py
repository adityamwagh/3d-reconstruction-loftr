import csv
import gc
import os
from glob import glob
from sched import scheduler

import cv2
import kornia
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from kornia_moons.feature import *

from loftr.loftr import LoFTR


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = LoFTR()

    def training_step(self):
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        return {"loss": loss, "log": {"loss": loss}}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    


model = Model()
torch.save(model.state_dict(), "weights/model_weights.ckpt")

    
