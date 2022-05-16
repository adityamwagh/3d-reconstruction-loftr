# !pip install ./kornia-loftr/kornia-0.6.4-py2.py3-none-any.whl
# !pip install ./kornia-loftr/kornia_moons-0.1.9-py3-none-any.whl
# !pip install ./loftrutils/einops-0.4.1-py3-none-any.whl


import sys

sys.path.append("./loftrutils/LoFTR-master/LoFTR-master/")


# !pip install loguru


from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path
import cv2
import torch
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt
import gc
from src.loftr import LoFTR
from src.loftr.utils.supervision import compute_supervision_coarse, compute_supervision_fine
from src.losses.loftr_loss import LoFTRLoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import compute_symmetrical_epipolar_errors, compute_pose_errors, aggregate_metrics
from src.utils.plotting import make_matching_figures
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler


class PL_LoFTR(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        self.loftr_cfg = lower_config(_config["loftr"])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = 1  # max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

        # Matcher: LoFTR
        self.matcher = LoFTR(config=_config["loftr"])
        self.loss = LoFTRLoss(_config)

        # Pretrained weights
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location="cuda")["state_dict"]
            self.matcher.load_state_dict(state_dict, strict=True)
            logger.info(f"Load '{pretrained_ckpt}' as pretrained checkpoint")

        # Testing
        self.dump_dir = dump_dir

    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == "linear":
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
            elif self.config.TRAINER.WARMUP_TYPE == "constant":
                pass
            else:
                raise ValueError(f"Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}")

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def _trainval_inference(self, batch):
        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)

        with self.profiler.profile("LoFTR"):
            self.matcher(batch)

        with self.profiler.profile("Compute fine supervision"):
            compute_supervision_fine(batch, self.config)

        with self.profiler.profile("Compute losses"):
            self.loss(batch)

    def _compute_metrics(self, batch):
        with self.profiler.profile("Copmute metrics"):
            compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
            compute_pose_errors(batch, self.config)  # compute R_errs, t_errs, pose_errs for each pair

            rel_pair_names = list(zip(*batch["pair_names"]))
            bs = batch["image0"].size(0)
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                "identifiers": ["#".join(rel_pair_names[b]) for b in range(bs)],
                "epi_errs": [batch["epi_errs"][batch["m_bids"] == b].cpu().numpy() for b in range(bs)],
                "R_errs": batch["R_errs"],
                "t_errs": batch["t_errs"],
                "inliers": batch["inliers"],
            }
            ret_dict = {"metrics": metrics}
        return ret_dict, rel_pair_names

    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)

        # logging
        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            # scalars
            for k, v in batch["loss_scalars"].items():
                self.logger.experiment.add_scalar(f"train/{k}", v, self.global_step)

            # net-params
            if self.config.LOFTR.MATCH_COARSE.MATCH_TYPE == "sinkhorn":
                self.logger.experiment.add_scalar(f"skh_bin_score", self.matcher.coarse_matching.bin_score.clone().detach().cpu().data, self.global_step)

            # figures
            if self.config.TRAINER.ENABLE_PLOTTING:
                compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
                figures = make_matching_figures(batch, self.config, self.config.TRAINER.PLOT_MODE)
                for k, v in figures.items():
                    self.logger.experiment.add_figure(f"train_match/{k}", v, self.global_step)
        gc.collect()
        torch.cuda.empty_cache()
        return {"loss": batch["loss"]}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar("train/avg_loss_on_epoch", avg_loss, global_step=self.current_epoch)
        gc.collect()
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        self._trainval_inference(batch)

        ret_dict, _ = self._compute_metrics(batch)

        val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        figures = {self.config.TRAINER.PLOT_MODE: []}
        if batch_idx % val_plot_interval == 0:
            figures = make_matching_figures(batch, self.config, mode=self.config.TRAINER.PLOT_MODE)
        gc.collect()
        torch.cuda.empty_cache()
        return {
            **ret_dict,
            "loss_scalars": batch["loss_scalars"],
            "figures": figures,
        }

    def validation_epoch_end(self, outputs):
        # handle multiple validation sets
        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        multi_val_metrics = defaultdict(list)

        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            if not self.trainer.resume_from_checkpoint:
                cur_epoch = -1

            # 1. loss_scalars: dict of list, on cpu
            _loss_scalars = [o["loss_scalars"] for o in outputs]
            loss_scalars = {k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars])) for k in _loss_scalars[0]}

            # 2. val metrics: dict of list, numpy
            _metrics = [o["metrics"] for o in outputs]
            metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
            # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
            for thr in [5, 10, 20]:
                multi_val_metrics[f"auc@{thr}"].append(val_metrics_4tb[f"auc@{thr}"])

            # 3. figures
            _figures = [o["figures"] for o in outputs]
            figures = {k: flattenList(gather(flattenList([_me[k] for _me in _figures]))) for k in _figures[0]}

            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    self.logger.experiment.add_scalar(f"val_{valset_idx}/avg_{k}", mean_v, global_step=cur_epoch)

                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch)

            gc.collect()
            torch.cuda.empty_cache()
            plt.close("all")

        for thr in [5, 10, 20]:
            # log on all ranks for ModelCheckpoint callback to work properly
            self.log(f"auc@{thr}", torch.tensor(np.mean(multi_val_metrics[f"auc@{thr}"])))  # ckpt monitors on this

    def test_step(self, batch, batch_idx):
        with self.profiler.profile("LoFTR"):
            self.matcher(batch)

        ret_dict, rel_pair_names = self._compute_metrics(batch)

        with self.profiler.profile("dump_results"):
            if self.dump_dir is not None:
                # dump results for further analysis
                keys_to_save = {"mkpts0_f", "mkpts1_f", "mconf", "epi_errs"}
                pair_names = list(zip(*batch["pair_names"]))
                bs = batch["image0"].shape[0]
                dumps = []
                for b_id in range(bs):
                    item = {}
                    mask = batch["m_bids"] == b_id
                    item["pair_names"] = pair_names[b_id]
                    item["identifier"] = "#".join(rel_pair_names[b_id])
                    for key in keys_to_save:
                        item[key] = batch[key][mask].cpu().numpy()
                    for key in ["R_errs", "t_errs", "inliers"]:
                        item[key] = batch[key][b_id]
                    dumps.append(item)
                ret_dict["dumps"] = dumps

        return ret_dict

    def test_epoch_end(self, outputs):
        # metrics: dict of list, numpy
        _metrics = [o["metrics"] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

        # [{key: [{...}, *#bs]}, *#batch]
        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o["dumps"] for o in outputs])  # [{...}, #bs*#batch]
            dumps = flattenList(gather(_dumps))  # [{...}, #proc*#bs*#batch]
            logger.info(f"Prediction and evaluation results will be saved to: {self.dump_dir}")

        if self.trainer.global_rank == 0:
            print(self.profiler.summary())
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
            logger.info("\n" + pprint.pformat(val_metrics_4tb))
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / "LoFTR_pred_eval", dumps)


def read_megadepth_depth(path, pad_to=None):
    depth = cv2.imread(path, 0)
    if pad_to is not None:
        depth, _ = pad_bottom_right(depth, pad_to, ret_mask=False)
    depth = torch.from_numpy(depth).float()  # (h, w)
    gc.collect()
    return depth


import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from loguru import logger

from src.utils.dataset import read_megadepth_gray, pad_bottom_right


class MegaDepthDataset(Dataset):
    def __init__(self, data, npz_path, mode="train", min_overlap_score=0.4, img_resize=None, df=None, img_padding=False, depth_padding=False, augment_fn=None, **kwargs):
        """
        Manage one scene(npz_path) of MegaDepth dataset.

        Args:
            root_dir (str): megadepth root directory that has `phoenix`.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            mode (str): options are ['train', 'val', 'test']
            min_overlap_score (float): how much a pair should have in common. In range of [0, 1]. Set to 0 when testing.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            depth_padding (bool): If set to 'True', zero-pad depthmap to (2000, 2000). This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()
        # self.root_dir = root_dir
        self.mode = mode

        # prepare scene_info and pair_info
        if mode == "test" and min_overlap_score != 0:
            logger.warning("You are using `min_overlap_score`!=0 in test mode. Set to 0.")
            min_overlap_score = 0

        # parameters for image resizing, padding and depthmap padding
        if mode == "train":
            assert img_resize is not None and img_padding and depth_padding
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        self.depth_max_size = 2000 if depth_padding else None  # the upperbound of depthmaps size in megadepth.

        # for training LoFTR
        self.augment_fn = augment_fn if mode == "train" else None
        self.coarse_scale = getattr(kwargs, "coarse_scale", 0.125)
        self.path1 = data["path1"].values
        self.path2 = data["path2"].values
        self.camerainst1 = data["camerainst1"].values
        self.camerainst2 = data["camerainst2"].values
        self.rot1 = data["rot1"].values
        self.rot2 = data["rot2"].values
        self.trans1 = data["trans1"].values
        self.trans2 = data["trans2"].values
        gc.collect()

    def __len__(self):
        return len(self.path1)

    def __getitem__(self, idx):
        # read grayscale image and mask. (1, h, w) and (h, w)
        img_name0 = self.path1[idx]
        img_name1 = self.path2[idx]

        # TODO: Support augmentation & handle seeds for each worker correctly.
        image0, mask0, scale0 = read_megadepth_gray(img_name0, self.img_resize, self.df, self.img_padding, None)
        # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        image1, mask1, scale1 = read_megadepth_gray(img_name1, self.img_resize, self.df, self.img_padding, None)
        # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        depth_path0 = "./depth-masks-imc2022/depth_maps/" + img_name0.split("/")[-3] + "/" + img_name0.split("/")[-1]
        depth_path1 = "./depth-masks-imc2022/depth_maps/" + img_name1.split("/")[-3] + "/" + img_name1.split("/")[-1]

        # read depth. shape: (h, w)
        if self.mode in ["train", "val"]:
            depth0 = read_megadepth_depth(depth_path0, pad_to=self.depth_max_size)
            depth1 = read_megadepth_depth(depth_path1, pad_to=self.depth_max_size)
        else:
            depth0 = depth1 = torch.tensor([])

        # read intrinsics of original size
        K_0 = torch.tensor(np.asarray([float(x) for x in self.camerainst1[idx].split(" ")]), dtype=torch.float).reshape(3, 3)
        K_1 = torch.tensor(np.asarray([float(x) for x in self.camerainst2[idx].split(" ")]), dtype=torch.float).reshape(3, 3)

        # read and compute relative poses
        R0 = self.rot1[idx].replace("{", "").replace("}", "").replace("'", "")
        R0 = np.asarray([float(x) for x in R0.split(" ")]).reshape(3, 3)
        Tv0 = self.trans1[idx].replace("{", "").replace("}", "").replace("'", "")
        Tv0 = np.asarray([[float(x) for x in Tv0.split(" ")]])
        T0 = np.concatenate((R0, Tv0.T), axis=1)
        T0 = np.concatenate((T0, np.asarray([[0, 0, 0, 1]])), axis=0)
        del R0
        del Tv0
        R1 = self.rot2[idx].replace("{", "").replace("}", "").replace("'", "")
        R1 = np.asarray([float(x) for x in R1.split(" ")]).reshape(3, 3)
        Tv1 = self.trans2[idx].replace("{", "").replace("}", "").replace("'", "")
        Tv1 = np.asarray([[float(x) for x in Tv1.split(" ")]])
        T1 = np.concatenate((R1, Tv1.T), axis=1)
        T1 = np.concatenate((T1, np.asarray([[0, 0, 0, 1]])), axis=0)
        del R1
        del Tv1
        T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]  # (4, 4)
        T_1to0 = T_0to1.inverse()

        data = {
            "image0": image0,  # (1, h, w)
            "depth0": depth0,  # (h, w)
            "image1": image1,
            "depth1": depth1,
            "T_0to1": T_0to1,  # (4, 4)
            "T_1to0": T_1to0,
            "K0": K_0,  # (3, 3)
            "K1": K_1,
            "scale0": scale0,  # [scale_w, scale_h]
            "scale1": scale1,
            "dataset_name": "MegaDepth",
            "scene_id": idx,
            "pair_id": idx,
            "pair_names": (img_name0, img_name1),
        }

        # for LoFTR training
        if mask0 is not None:  # img_padding is True
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(
                    torch.stack([mask0, mask1], dim=0)[None].float(), scale_factor=self.coarse_scale, mode="nearest", recompute_scale_factor=False
                )[0].bool()
            data.update({"mask0": ts_mask_0, "mask1": ts_mask_1})
        del image0
        del image1
        del depth0
        del depth1
        gc.collect()
        torch.cuda.empty_cache()
        return data


import os
import math
from collections import abc
from loguru import logger
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from os import path as osp
from pathlib import Path
from joblib import Parallel, delayed

import pytorch_lightning as pl
from torch import distributed as dist
from torch.utils.data import Dataset, DataLoader, ConcatDataset, DistributedSampler, RandomSampler, dataloader

from src.utils.augment import build_augmentor
from src.utils.dataloader import get_local_split
from src.utils.misc import tqdm_joblib
from src.utils import comm


from src.datasets.sampler import RandomConcatSampler


class MultiSceneDataModule(pl.LightningDataModule):
    """
    For distributed training, each training process is assgined
    only a part of the training scenes to reduce memory overhead.
    """

    def __init__(self, args, config, data):
        super().__init__()

        # 1. data config
        # Train and Val should from the same data source
        self.trainval_data_source = config.DATASET.TRAINVAL_DATA_SOURCE
        self.test_data_source = config.DATASET.TEST_DATA_SOURCE
        # training and validating
        self.train_data = data
        self.train_pose_root = config.DATASET.TRAIN_POSE_ROOT  # (optional)
        self.train_npz_root = config.DATASET.TRAIN_NPZ_ROOT
        self.train_list_path = config.DATASET.TRAIN_LIST_PATH
        self.train_intrinsic_path = config.DATASET.TRAIN_INTRINSIC_PATH
        self.val_data = data
        self.val_pose_root = config.DATASET.VAL_POSE_ROOT  # (optional)
        self.val_npz_root = config.DATASET.VAL_NPZ_ROOT
        self.val_list_path = config.DATASET.VAL_LIST_PATH
        self.val_intrinsic_path = config.DATASET.VAL_INTRINSIC_PATH
        # testing
        self.test_data = data
        self.test_pose_root = config.DATASET.TEST_POSE_ROOT  # (optional)
        self.test_npz_root = config.DATASET.TEST_NPZ_ROOT
        self.test_list_path = config.DATASET.TEST_LIST_PATH
        self.test_intrinsic_path = config.DATASET.TEST_INTRINSIC_PATH

        # 2. dataset config
        # general options
        self.min_overlap_score_test = config.DATASET.MIN_OVERLAP_SCORE_TEST  # 0.4, omit data with overlap_score < min_overlap_score
        self.min_overlap_score_train = config.DATASET.MIN_OVERLAP_SCORE_TRAIN
        self.augment_fn = build_augmentor(config.DATASET.AUGMENTATION_TYPE)  # None, options: [None, 'dark', 'mobile']

        # MegaDepth options
        self.mgdpt_img_resize = config.DATASET.MGDPT_IMG_RESIZE  # 840
        self.mgdpt_img_pad = config.DATASET.MGDPT_IMG_PAD  # True
        self.mgdpt_depth_pad = config.DATASET.MGDPT_DEPTH_PAD  # True
        self.mgdpt_df = config.DATASET.MGDPT_DF  # 8
        self.coarse_scale = 1 / config.LOFTR.RESOLUTION[0]  # 0.125. for training loftr.

        # 3.loader parameters
        self.train_loader_params = {"batch_size": args.batch_size, "num_workers": args.num_workers, "pin_memory": getattr(args, "pin_memory", True)}
        self.val_loader_params = {"batch_size": 1, "shuffle": False, "num_workers": args.num_workers, "pin_memory": getattr(args, "pin_memory", True)}
        self.test_loader_params = {"batch_size": 1, "shuffle": False, "num_workers": args.num_workers, "pin_memory": True}

        # 4. sampler
        self.data_sampler = config.TRAINER.DATA_SAMPLER
        self.n_samples_per_subset = config.TRAINER.N_SAMPLES_PER_SUBSET
        self.subset_replacement = config.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT
        self.shuffle = config.TRAINER.SB_SUBSET_SHUFFLE
        self.repeat = config.TRAINER.SB_REPEAT

        # (optional) RandomSampler for debugging

        # misc configurations
        self.parallel_load_data = getattr(args, "parallel_load_data", False)
        self.seed = config.TRAINER.SEED  # 66

    def setup(self, stage=None):
        """
        Setup train / val / test dataset. This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase, and 'test' in testing phase.
        """

        assert stage in ["fit", "test"], "stage must be either fit or test"

        if stage == "fit":
            self.train_dataset = self._setup_dataset(
                self.train_data,
                self.train_npz_root,
                self.train_list_path,
                self.train_intrinsic_path,
                mode="train",
                min_overlap_score=self.min_overlap_score_train,
                pose_dir=self.train_pose_root,
            )
            # setup multiple (optional) validation subsets

            self.val_dataset = self._setup_dataset(
                self.val_data,
                self.val_npz_root,
                self.val_list_path,
                self.val_intrinsic_path,
                mode="val",
                min_overlap_score=self.min_overlap_score_test,
                pose_dir=self.val_pose_root,
            )

        else:  # stage == 'test
            self.test_dataset = self._setup_dataset(
                self.test_data,
                self.test_npz_root,
                self.test_list_path,
                self.test_intrinsic_path,
                mode="test",
                min_overlap_score=self.min_overlap_score_test,
                pose_dir=self.test_pose_root,
            )

    def _setup_dataset(self, data, split_npz_root, scene_list_path, intri_path, mode="train", min_overlap_score=0.0, pose_dir=None):
        """Setup train / val / test set"""
        local_npz_names = ""
        dataset_builder = self._build_concat_dataset
        return dataset_builder(data, local_npz_names, split_npz_root, intri_path, mode=mode, min_overlap_score=min_overlap_score, pose_dir=pose_dir)

    def _build_concat_dataset(self, data, npz_names, npz_dir, intrinsic_path, mode, min_overlap_score=0.0, pose_dir=None):
        datasets = []
        augment_fn = self.augment_fn if mode == "train" else None
        data_source = self.trainval_data_source if mode in ["train", "val"] else self.test_data_source
        npz_path = ""

        datasets.append(
            MegaDepthDataset(
                data,
                npz_path,
                mode=mode,
                min_overlap_score=min_overlap_score,
                img_resize=self.mgdpt_img_resize,
                df=self.mgdpt_df,
                img_padding=self.mgdpt_img_pad,
                depth_padding=self.mgdpt_depth_pad,
                augment_fn=augment_fn,
                coarse_scale=self.coarse_scale,
            )
        )
        return ConcatDataset(datasets)

    def train_dataloader(self):
        """Build training dataloader for ScanNet / MegaDepth."""
        #         assert self.data_sampler in ['scene_balance']
        #         #logger.info(f'[rank:{self.rank}/{self.world_size}]: Train Sampler and DataLoader re-init (should not re-init between epochs!).')
        #         if self.data_sampler == 'scene_balance':
        #             sampler = RandomConcatSampler(self.train_dataset,
        #                                           self.n_samples_per_subset,
        #                                           self.subset_replacement,
        #                                           self.shuffle, self.repeat, self.seed)
        #         else:
        #             sampler = None
        dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
        return dataloader

    def val_dataloader(self):
        """Build validation dataloader for ScanNet / MegaDepth."""
        # logger.info(f'[rank:{self.rank}/{self.world_size}]: Val Sampler and DataLoader re-init.')
        dataloader = DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
        return dataloader

    def test_dataloader(self, *args, **kwargs):
        # logger.info(f'[rank:{self.rank}/{self.world_size}]: Test Sampler and DataLoader re-init.')
        sampler = DistributedSampler(self.test_dataset, shuffle=False)
        return DataLoader(self.test_dataset, sampler=sampler, **self.test_loader_params)


def _build_dataset(dataset: Dataset, *args, **kwargs):
    return dataset(*args, **kwargs)


import math
import argparse
import pprint
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin

from src.config.default import get_cfg_defaults
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.utils.profiler import build_profiler
import pandas as pd

loguru_logger = get_rank_zero_only_logger(loguru_logger)


def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_cfg_path", type=str, help="data config path")
    parser.add_argument("main_cfg_path", type=str, help="main config path")
    parser.add_argument("--exp_name", type=str, default="default_exp_name")
    parser.add_argument("--batch_size", type=int, default=4, help="batch_size per gpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=lambda x: bool(strtobool(x)), nargs="?", default=True, help="whether loading data to pinned memory or not")
    parser.add_argument("--ckpt_path", type=str, default="./kornia-loftr/outdoor_ds.ckpt", help="pretrained checkpoint path, helpful for using a pre-trained coarse-only LoFTR")
    parser.add_argument("--disable_ckpt", action="store_true", help="disable checkpoint saving (useful for debugging).")
    parser.add_argument("--profiler_name", type=str, default=None, help="options: [inference, pytorch], or leave it unset")
    parser.add_argument("--parallel_load_data", action="store_true", help="load datasets in with multiple processes.")

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args(
        "./loftrutils/LoFTR-master/LoFTR-master/configs/data/megadepth_trainval_640.py ./loftrutils/LoFTR-master/LoFTR-master/configs/loftr/outdoor/loftr_ds_dense.py --exp_name test --gpus 0 --num_nodes 0 --accelerator gpu --batch_size 1 --check_val_every_n_epoch 1 --log_every_n_steps 1 --flush_logs_every_n_steps 1 --limit_val_batches 1 --num_sanity_val_steps 10 --benchmark True --max_epochs 4".split()
    )


def train():
    # parse arguments
    args = parse_args()
    rank_zero_only(pprint.pprint)(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility
    # TODO: Use different seeds for each dataloader workers
    # This is needed for data augmentation

    # scale lr and warmup-step automatically
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = 1  # config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = 0.00001 * _scaling
    config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP / _scaling)

    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_LoFTR(config, pretrained_ckpt=args.ckpt_path, profiler=profiler)
    loguru_logger.info(f"LoFTR LightningModule initialized!")

    # lightning data
    data = pd.read_csv("./imc-gt/train.csv")
    data_module = MultiSceneDataModule(args, config, data[:100])
    gc.collect()
    loguru_logger.info(f"LoFTR DataModule initialized!")

    # TensorBoard Logger
    logger = TensorBoardLogger(save_dir="logs/tb_logs", name=args.exp_name, default_hp_metric=False)
    ckpt_dir = Path(logger.log_dir) / "checkpoints"

    # Callbacks
    # TODO: update ModelCheckpoint to monitor multiple metrics
    ckpt_callback = ModelCheckpoint(
        monitor="auc@10", verbose=True, save_top_k=5, mode="max", save_last=True, dirpath=str(ckpt_dir), filename="{epoch}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [lr_monitor]
    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)

    # Lightning Trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        #         plugins=DDPPlugin(find_unused_parameters=False,
        #                           num_nodes=args.num_nodes,
        #                           sync_batchnorm=config.TRAINER.WORLD_SIZE > 0),
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        callbacks=callbacks,
        logger=logger,
        # sync_batchnorm=config.TRAINER.WORLD_SIZE > 0,
        replace_sampler_ddp=False,  # use custom sampler
        # avoid repeated samples!
        weights_summary="full",
        profiler=profiler,
    )
    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start training!")
    trainer.fit(model, datamodule=data_module)


train()
