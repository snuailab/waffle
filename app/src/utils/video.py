from pathlib import Path
from typing import Union

import cv2
import tqdm
from waffle_utils.file import io


def video_frames_save(
    video_path: Union[str, Path], save_path: Union[str, Path], capture_frame_rate: float = 1.0
):
    """Save frames from a video file.
    Args:
        video_path (Union[str, Path]): Path to the video file.
        save_path (Union[str, Path]): Path to save the frames.
        capture_frame_rate (float, optional): Frame capture rate. Defaults to 1.0.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = round(fps / capture_frame_rate)

    if not Path(save_path).exists():
        io.make_directory(save_path)

    frame_idxs = list(range(0, total_frames, interval))
    for i in tqdm.tqdm(frame_idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(Path(save_path) / f"{Path(video_path).stem}_{i}.jpg"), frame)

    cap.release()
