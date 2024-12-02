import collections
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision
from torchvision.transforms import v2

# Global variable for directory paths
dir2 = os.path.abspath("/volume/DeepCORO_CLIP/orion")
dir1 = os.path.dirname(dir2)    
if dir1 not in sys.path:
    sys.path.append(dir1)

import utils.data_processing


class Video(torch.utils.data.Dataset):
    """
    A dataset class for handling orion video data.

    Args:
        root (str): Root directory of the dataset. Defaults to '../../data/'.
        data_filename (str): Name of the data file. Defaults to None.
        split (str): Dataset split to use ('train', 'val', 'test'). Defaults to 'train'.
        target_label (str): Name of the target label column. Defaults to None.
        datapoint_loc_label (str): Column name for file paths. Defaults to 'FileName'.
        -ize (int): Size to resize videos to (height, width). Defaults to 224.
        mean (float or np.array): Channel-wise means for normalization. Defaults to 0.0.
        std (float or np.array): Channel-wise stds for normalization. Defaults to 1.0.
        length (int): Number of frames to clip from the video. Defaults to 32.
        period (int): Sampling period for frame clipping. Defaults to 1.
        max_length (int): Max number of frames to clip. Defaults to 250.
        pad (int): Number of pixels for padding frames. Defaults to None.
        noise (float): Fraction of pixels to black out as noise. Defaults to None.
        video_transforms (list): List of torchvision transforms. Defaults to None.
        rand_augment (bool): Apply random augmentation. Defaults to False.
        apply_mask (bool): Apply masking to videos. Defaults to False.
        target_transform (callable): Transform to apply to the target. Defaults to None.
        external_test_location (str): Path for external testing videos. Defaults to None.

    Returns:
        Tuple: normalized, resized, and augmented videos with labels.

    Note:
        Augmentations, normalizations, frame sampling, and resizing have been
        rigorously tested and visualized.
    """

    def __init__(
        self,
        root="../../data/",
        data_filename=None,
        split="train",
        target_label=None,
        datapoint_loc_label="FileName",
        resize=224,
        mean=0.0,
        std=1.0,
        length=32,
        period=1,
        max_length=250,
        clips=1,
        pad=None,
        noise=None,
        video_transforms=None,
        rand_augment=False,
        apply_mask=False,
        target_transform=None,
        external_test_location=None,
        weighted_sampling=False,
        normalize=True,
        debug=False,
    ) -> None:
        # Initialize instance variables
        self.folder = pathlib.Path(root)
        self.filename = data_filename
        self.datapoint_loc_label = datapoint_loc_label
        self.split = split
        if not isinstance(target_label, list):
            target_label = [target_label]
        self.target_label = target_label
        self.mean = format_mean_std(mean)
        self.std = format_mean_std(std)
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.video_transforms = video_transforms
        self.rand_augment = rand_augment
        self.apply_mask = apply_mask
        self.target_transform = target_transform
        self.external_test_location = external_test_location
        self.resize = resize
        self.weighted_sampling = weighted_sampling
        self.debug = debug
        self.normalize = normalize

        self.fnames, self.outcome = [], []
        if split == "external_test":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            with open(os.path.join(self.folder, self.filename)) as f:
                self.header = f.readline().strip().split("α")
                if len(self.header) == 1:
                    raise ValueError(
                        "Header was not split properly. Please ensure the file uses 'α' (alpha) as the delimiter."
                    )

            self.fnames, self.outcomes, target_index = self.load_data(split, target_label)
            self.frames = collections.defaultdict(list)
            # pdb.set_trace()
            self.trace = collections.defaultdict(_defaultdict_of_lists)

        if self.weighted_sampling is True:
            # define weights for weighted sampling
            labels = np.array(
                [self.outcome[ind][target_index] for ind in range(len(self.outcome))], dtype=int
            )

            # binary weights length == 2
            weights = 1 - (np.bincount(labels) / len(labels))
            self.weight_list = np.zeros(len(labels))

            for label in range(len(weights)):
                weight = weights[label]
                self.weight_list[np.where(labels == label)] = weight

    def load_data(self, split, target_label):
        # Read the "α" separated file using pandas
        file_path = os.path.join(self.folder, self.filename)
        data = pd.read_csv(file_path, sep="α", engine="python")

        filename_index = data.columns.get_loc(self.datapoint_loc_label)
        split_index = data.columns.get_loc("Split")
        if target_label is None:
            target_index = None
        else:
            target_index = data.columns.get_loc(target_label[0])

        self.fnames = []
        self.outcome = []
        # Iterate through rows using iterrows
        for index, row in data.iterrows():
            file_name = row.iloc[filename_index]
            file_mode = row.iloc[split_index].lower()

            if split in ["all", file_mode] and os.path.exists(file_name):
                self.fnames.append(file_name)
                self.outcome.append(row.iloc[target_index])

        return self.fnames, self.outcome, target_index

    def __getitem__(self, index):
        # Find filename of video
        if self.split == "external_test":
            video_fname = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "clinical_test":
            video_fname = os.path.join(self.folder, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video_fname = self.fnames[index]

        video = utils.data_processing.loadvideo(video_fname).astype(np.float32)

        if self.apply_mask:
            path = video_fname.rsplit("/", 2)
            mask_filename = f"{path[0]}/mask/{path[2]}"
            mask_filename = mask_filename.split(".avi")[0] + ".npy"

            mask = np.load(mask_filename).transpose(2, 0, 1)

            # fix mask shapes
            length = video.shape[2]
            if mask.shape[1] < length:
                mask = np.pad(mask, [(0, 0), (length - mask.shape[1], 0), (0, 0)])
            if mask.shape[2] < length:
                mask = np.pad(mask, [(0, 0), (0, 0), (length - mask.shape[2], 0)])
            if mask.shape[1] > length:
                mask = mask[:, :length, :]
            if mask.shape[2] > length:
                mask = mask[:, :, :length]

            for ind in range(video.shape[0]):
                video[ind, :, :, :] = video[ind, :, :, :] * mask

        # Add simulated noise (black out random pixels)
        # 0 represents black at this point (video has not been normalized yet)
        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0

        video = torch.from_numpy(video)
        if self.resize is not None:
            video = v2.Resize((self.resize, self.resize), antialias=True)(video)

        if self.normalize == True:
            if hasattr(self, "mean") and hasattr(self, "std"):
                video = v2.Normalize(self.mean, self.std)(video)

        if self.video_transforms is not None:
            transforms = v2.RandomApply(torch.nn.ModuleList(self.video_transforms), p=0.5)
            scripted_transforms = torch.jit.script(transforms)
            try:
                video = scripted_transforms(video)
            except RuntimeError as e:
                print(f"Skipping video {self.fnames[index]} due to error: {str(e)}")
                return self.__getitem__(index + 1)  # retry with next sample

        if self.rand_augment:
            raug = [
                v2.RandAugment(
                    magnitude=9,
                    num_layers=2,
                    prob=0.5,
                    sampling_type="gaussian",
                    sampling_hparas=None,
                )
            ]
            raug_composed = v2.Compose(raug)
            video = raug_composed(video)

        # Permute the tensor to have the shape [F, H, W, C]

        video = video.permute(1, 0, 2, 3)
        video = video.numpy()

        # Set number of frames
        c, f, h, w = video.shape
        length = f // self.period if self.length is None else self.length

        if self.max_length is not None:
            # Shorten videos to max_length
            length = min(length, self.max_length)

        if f < length * self.period:
            # Pad video with frames filled with zeros if too short
            # 0 represents the mean color (dark grey), since this is after normalization
            video = np.concatenate(
                (video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1
            )
            c, f, h, w = video.shape  # pylint: disable=E0633

        if self.clips == "all":
            # Take all possible clips of desired length
            start = np.arange(f - (length - 1) * self.period)
        else:
            # Take random clips from video
            start = np.random.choice(f - (length - 1) * self.period, self.clips)

        if self.target_label is not None:
            # Gather targets
            target = []
            for t in self.target_label:
                key = os.path.splitext(self.fnames[index])[0]
                if t == "Filename":
                    target.append(self.fnames[index])
                elif t == "LargeIndex":
                    # Traces are sorted by cross-sectional area
                    # Largest (diastolic) frame is last
                    target.append(np.int(self.frames[key][-1]))
                elif t == "SmallIndex":
                    # Largest (diastolic) frame is first
                    target.append(np.int(self.frames[key][0]))
                elif t == "LargeFrame":
                    target.append(video[:, self.frames[key][-1], :, :])
                elif t == "SmallFrame":
                    target.append(video[:, self.frames[key][0], :, :])
                elif t in ["LargeTrace", "SmallTrace"]:
                    if t == "LargeTrace":
                        t = self.trace[key][self.frames[key][-1]]
                    else:
                        t = self.trace[key][self.frames[key][0]]
                    x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                    x = np.concatenate((x1[1:], np.flip(x2[1:])))
                    y = np.concatenate((y1[1:], np.flip(y2[1:])))

                    r, c = skimage.draw.polygon(
                        np.rint(y).astype(np.int),
                        np.rint(x).astype(np.int),
                        (video.shape[2], video.shape[3]),
                    )
                    mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                    mask[r, c] = 1
                    target.append(mask)
                else:
                    # change this
                    #     target.append(np.float32(0))
                    # else:
                    #     target.append(np.float32(self.outcome[index][self.header.index(t)]))
                    # change this
                    # print(self.outcome)
                    # print(self.outcome[index])
                    # print(t)
                    # print(self.header)
                    # print(self.header.index(t))
                    target.append(self.outcome[index])

                target = [np.float32(i) for i in target]
        if target is not None and target != []:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target.cuda())

        # Select random clips
        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
        video = video[0] if self.clips == 1 else np.stack(video)
        if self.pad is not None:
            # Add padding of zeros (mean color of videos)
            # Crop of original size is taken out
            # (Used as augmentation)
            c, l, h, w = video.shape
            temp = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp[
                :, :, self.pad : -self.pad, self.pad : -self.pad
            ] = video  # pylint: disable=E1130
            i, j = np.random.randint(0, 2 * self.pad, 2)
            video = temp[:, :, i : (i + h), j : (j + w)]

        if self.split == "inference":
            return video, self.fnames[index]

        return video, target, video_fname

    def __len__(self):
        return len(self.fnames)


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.
    This is used to avoid issues with Windows (if this function is anonymous,
    the Video dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)


class Video_Multi(torch.utils.data.Dataset):
    """orion Dataset.
    Args:
        root (string): Root directory of dataset (defaults to `orion.config.DATA_DIR`)
        split (string): One of {"train", "val", "test", "external_test"}
        target_label (string or list, optional): Type of target to use,
            ``Filename'', ``EF'', ``EDV'', ``ESV'', ``LargeIndex'',
            ``SmallIndex'', ``LargeFrame'', ``SmallFrame'', ``LargeTrace'',
            or ``SmallTrace''
            Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``Filename'' (string): filename of video
                ``EF'' (float): ejection fraction
                ``EDV'' (float): end-diastolic volume
                ``ESV'' (float): end-systolic volume
                ``LargeIndex'' (int): index of large (diastolic) frame in video
                ``SmallIndex'' (int): index of small (systolic) frame in video
                ``LargeFrame'' (np.array shape=(3, height, width)): normalized large (diastolic) frame
                ``SmallFrame'' (np.array shape=(3, height, width)): normalized small (systolic) frame
                ``LargeTrace'' (np.array shape=(height, width)): left ventricle large (diastolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
                ``SmallTrace'' (np.array shape=(height, width)): left ventricle small (systolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
            Defaults to ``EF''.
        mean (int, float, or np.array shape=(3,), optional): means for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not shifted).
        std (int, float, or np.array shape=(3,), optional): standard deviation for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not scaled).
        length (int or None, optional): Number of frames to clip from video. If ``None'', longest possible clip is returned.
            Defaults to 16.
        period (int, optional): Sampling period for taking a clip from the video (i.e. every ``period''-th frame is taken)
            Defaults to 2.
        max_length (int or None, optional): Maximum number of frames to clip from video (main use is for shortening excessively
            long videos when ``length'' is set to None). If ``None'', shortening is not applied to any video.
            Defaults to 250.
        clips (int, optional): Number of clips to sample. Main use is for test-time augmentation with random clips.
            Defaults to 1.
        pad (int or None, optional): Number of pixels to pad all frames on each side (used as augmentation).
            and a window of the original size is taken. If ``None'', no padding occurs.
            Defaults to ``None''.
        noise (float or None, optional): Fraction of pixels to black out as simulated noise. If ``None'', no simulated noise is added.
            Defaults to ``None''.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        external_test_location (string): Path to videos to use for external testing.
    """

    def __init__(
        self,
        root="../../data/",
        data_filename=None,
        split="train",
        target_label="HCM",
        datapoint_loc_label="FileName",
        resize=224,
        view_count=2,
        mean=0.0,
        std=1.0,
        length=32,
        period=1,
        max_length=250,
        clips=1,
        pad=None,
        noise=None,
        video_transforms=None,
        apply_mask=False,
        rand_augment=False,
        num_classes=None,
        target_transform=None,
        external_test_location=None,
        weighted_sampling=False,
        debug=False,
        normalize=True,
    ) -> None:
        self.folder = pathlib.Path(root)
        self.filename = data_filename
        self.datapoint_loc_label = datapoint_loc_label
        self.split = split
        # if not isinstance(target_label, list):
        #     target_label = [target_label]
        self.target_label = target_label
        self.mean = format_mean_std(mean)
        self.std = format_mean_std(std)
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.video_transforms = video_transforms
        self.apply_mask = apply_mask
        self.rand_augment = rand_augment
        self.target_transform = target_transform
        self.external_test_location = external_test_location
        self.resize = resize
        self.view_count = view_count
        self.debug = debug
        self.weighted_sampling = weighted_sampling
        self.normalize = normalize

        self.fnames, self.outcome = [], []

        df_dataset = pd.read_csv(
            os.path.join(self.folder, self.filename), sep="α", engine="python"
        )

        filenameIndex = df_dataset.columns.get_loc(self.datapoint_loc_label + str(0))
        splitIndex = df_dataset.columns.get_loc("Split")
        target_index = df_dataset.columns.get_loc(target_label)
        for _, row in df_dataset.iterrows():
            view_count = int(self.view_count)
            filenameIndex = [
                df_dataset.columns.get_loc(self.datapoint_loc_label + str(i))
                for i in range(view_count)
            ]
            outcomeIndex = [
                df_dataset.columns.get_loc(self.target_label) for _ in range(view_count)
            ]
            fileMode = row.iloc[splitIndex].lower()
            fileName = [row.iloc[i] for i in filenameIndex]
            outcomes = [row.iloc[i] for i in outcomeIndex]

            ##Append all files to file_vids array
            file_vids = [
                i for i in fileName if self.split in ["all", fileMode] and os.path.exists(i)
            ]
            if len(file_vids) > 1:
                self.fnames.append(file_vids)
                self.outcome.append(outcomes)
        if self.debug:
            print(self.fnames)
            print(len(self.fnames))
            print(self.outcome)
            print(len(self.outcome))

        self.frames = collections.defaultdict(list)
        self.trace = collections.defaultdict(_defaultdict_of_lists)

        if self.weighted_sampling is True:
            # define weights for weighted sampling
            labels = np.array(
                [self.outcome[ind][target_index] for ind in range(len(self.outcome))], dtype=int
            )

            # binary weights length == 2
            weights = 1 - (np.bincount(labels) / len(labels))
            self.weight_list = np.zeros(len(labels))

            for label in range(len(weights)):
                weight = weights[label]
                self.weight_list[np.where(labels == label)] = weight

    #         # define weights for weighted sampling
    #         labels = np.array([self.outcome[ind][target_index] for ind in range(len(self.outcome))], dtype=int)
    #         weights = 1 - (np.bincount(labels) / len(labels))

    #         self.weight_list = np.zeros(len(labels))

    #         for label in range(len(weights)):
    #             weight = weights[label]
    #             self.weight_list[np.where(labels == label)] = weight

    def make_video(self, video, count, index):
        video = utils.data_processing.loadvideo(video).astype(np.float32)

        if self.apply_mask:
            path = video_fname.rsplit("/", 2)
            mask_filename = f"{path[0]}/mask/{path[2]}"
            mask_filename = mask_filename.split(".avi")[0] + ".npy"

            mask = np.load(mask_filename).transpose(2, 0, 1)

            # fix mask shapes
            length = video.shape[2]
            if mask.shape[1] < length:
                mask = np.pad(mask, [(0, 0), (length - mask.shape[1], 0), (0, 0)])
            if mask.shape[2] < length:
                mask = np.pad(mask, [(0, 0), (0, 0), (length - mask.shape[2], 0)])
            if mask.shape[1] > length:
                mask = mask[:, :length, :]
            if mask.shape[2] > length:
                mask = mask[:, :, :length]

            for ind in range(video.shape[0]):
                video[ind, :, :, :] = video[ind, :, :, :] * mask
        # Add simulated noise (black out random pixels)
        # 0 represents black at this point (video has not been normalized yet)
        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0

        video = torch.from_numpy(video)
        # if self.debug is True:
        #    self.plot_middle_frame(video, "original", index)

        if self.resize is not None:
            video = v2.Resize((self.resize, self.resize), antialias=True)(video)
        #       apply normalization before augmentaiton per riselab and torchvision
        #       first initialization is a float of 0 so need to avoid indexing a float

        #       apply normalization before augmentaiton per riselab and torchvision

        # If self.mean and self.std are defined, apply normalization
        if self.normalize == True:
            if hasattr(self, "mean") and hasattr(self, "std"):
                video = v2.Normalize(self.mean, self.std)(video)

        # if self.debug is True:
        #    self.plot_middle_frame(video, "after normalize", index)
        if self.video_transforms is not None:
            transforms = v2.RandomApply(torch.nn.ModuleList(self.video_transforms), p=0.5)
            scripted_transforms = torch.jit.script(transforms)
            video = scripted_transforms(video)

        if self.rand_augment:
            raug = [v2.RandAugment(magnitude=9, num_ops=2)]

            raug_composed = v2.Compose(raug)
            video = video.to(torch.uint8)
            video = raug_composed(video)
        # if self.debug is True:
        #   self.plot_middle_frame(video, "after random augment", index)
        video = video.permute(1, 0, 2, 3)
        video = video.numpy()

        # Set number of frames
        c, f, h, w = video.shape
        length = f // self.period if self.length is None else self.length
        if self.max_length is not None:
            # Shorten videos to max_length
            length = min(length, self.max_length)

        if f < length * self.period:
            # Pad video with frames filled with zeros if too short
            # 0 represents the mean color (dark grey), since this is after normalization
            video = np.concatenate(
                (video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1
            )
            c, f, h, w = video.shape  # pylint: disable=E0633

        if self.clips == "all":
            # Take all possible clips of desired length
            start = np.arange(f - (length - 1) * self.period)
        else:
            # Take random clips from video
            start = np.random.choice(f - (length - 1) * self.period, self.clips)

        # Select random clips
        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
        video = video[0] if self.clips == 1 else np.stack(video)

        if self.pad is not None:
            # Add padding of zeros (mean color of videos)
            # Crop of original size is taken out
            # (Used as augmentation)
            c, l, h, w = video.shape
            temp = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp[
                :, :, self.pad : -self.pad, self.pad : -self.pad
            ] = video  # pylint: disable=E1130
            i, j = np.random.randint(0, 2 * self.pad, 2)
            video = temp[:, :, i : (i + h), j : (j + w)]

        return video

    def make_targets(self, index):
        # Gather targets
        target = []
        for t in self.target_label:
            if t == "Filename1":
                target.append(self.fnames[index][1])
            elif self.split in ["clinical_test", "external_test"]:
                target.append(np.float32(0))
            else:
                target.append(np.float32(self.outcome[index]))

            if target != []:
                target = tuple(target) if len(target) > 1 else target[0]
                if self.target_transform is not None:
                    target = self.target_transform(target)

            return target

    def __getitem__(self, index):
        # print(f"Loading data at index: {index}")
        if self.split == "external_test":
            for i in self.fnames[index][0]:
                video = os.path.join(self.external_test_location, i)
        elif self.split == "clinical_test":
            for i in self.fnames[index][0]:
                video = os.path.join(self.folder, "ProcessedStrainStudyA4c", i)
        else:
            vids_stack = []
            filenames = []
            if self.split == "inference":
                for count, i in enumerate(self.fnames[index]):
                    video = Video_Multi.make_video(self, i, count, index)
                    vids_stack.append(video)
                    filenames.append(i)
                return vids_stack, filenames
            else:
                targets = []
                # print(self.fnames[index])
                for count, i in enumerate(self.fnames[index]):
                    # print(len(self.fnames[index]))
                    # print(self.fnames[index])
                    video = Video_Multi.make_video(self, i, count, index)
                    vids_stack.append(video)
                    target = Video_Multi.make_targets(self, index)
                    targets.append(target)
                    filenames.append(i)

                # print("Targets 1,1", targets[1][1])
                # print("Vids_stack", len(vids_stack))
                # print("Filenames Video.py", filenames)

                return vids_stack, targets[1][1], filenames

                # return vids_stack, targets[1][1]

    def __len__(self):
        return len(self.fnames)

    def plot_middle_frame(self, x, title, index):
        # Only plot for every 10th example
        if index % 10 == 0:
            # Calculate the index of the middle frame
            print(f"Min: {x.min()}, Max: {x.max()}, Title: {title}")

            middle_frame_index = x.shape[0] // 2

            # Select the middle frame
            frame = x[middle_frame_index, :, :, :]

            # Convert the numpy array to a PyTorch tensor if it's not already a tensor
            if not isinstance(frame, torch.Tensor):
                frame = torch.from_numpy(frame)

            # Permute the tensor dimensions
            frame = frame.permute(1, 2, 0)

            # Convert the tensor to a numpy array
            frame = frame.cpu().numpy()

            # Normalize the pixel values to the range [0, 1] for plotting
            frame = (frame - frame.min()) / (frame.max() - frame.min())

            # Plot the frame
            plt.imshow(frame)
            plt.title(title)
            plt.axis("off")
            plt.show()


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.
    This is used to avoid issues with Windows (if this function is anonymous,
    the Video dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)


def format_mean_std(input_value):
    """
    Formats the mean or std value to a list of floats.

    Args:
        input_value (str, list, np.array): The input mean/std value.

    Returns:
        list: A list of floats.

    Raises:
        ValueError: If the input cannot be converted to a list of floats.
    """

    if input_value is 1.0 or input_value is 0.0:
        print("Mean/STD value is not defined")
        return input_value

    # Check if input is a list with a single string element
    if (
        isinstance(input_value, list)
        and len(input_value) == 1
        and isinstance(input_value[0], str)
    ):
        # Extract the string from the list and process it
        input_value = input_value[0]

    if isinstance(input_value, str):
        # Remove any brackets or extra whitespace and split the string
        cleaned_input = (
            input_value.replace("[", "").replace("]", "").strip().replace("’", "").split()
        )
        try:
            # Convert the split string values to float
            formatted_value = [float(val) for val in cleaned_input]
        except ValueError:
            raise ValueError("String input for mean/std must be space-separated numbers.")
    elif isinstance(input_value, (list, np.ndarray)):
        try:
            # Convert elements to float
            formatted_value = [float(val) for val in input_value]
        except ValueError:
            raise ValueError("List or array input for mean/std must contain numbers.")
    else:
        raise TypeError("Input for mean/std must be a string, list, or numpy array.")

    if len(formatted_value) != 3:
        raise ValueError("Mean/std must have exactly three elements (for RGB channels).")

    return formatted_value
