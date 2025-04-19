import os

import cv2
import numpy as np

from metric import fvd


def generate_my_video_path_list(eval_name):
    # Get PWD of the dir where the script is located
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_path = os.path.join(current_dir, 'samples')

    base_path = os.path.join(sample_path, 'base')
    eval_path = os.path.join(sample_path, eval_name)

    video_names = []
    
    for video_name in os.listdir(base_path):
        if video_name.endswith('.mp4'):
            video_names.append(video_name)

    video_names.sort()

    # Sanity check, make sure the number of videos is the same
    for video_name in video_names:
        if not os.path.exists(os.path.join(eval_path, video_name)):
            raise ValueError(f"Video {video_name} not found in {eval_path}")

    base_video_paths = []
    eval_video_paths = []
    for video_name in video_names:
        base_video_paths.append(os.path.join(base_path, video_name))
        eval_video_paths.append(os.path.join(eval_path, video_name))

    return video_names, base_video_paths, eval_video_paths


def cv2_loadvideo(video_path):

    cap = cv2.VideoCapture(video_path)

    assert cap.isOpened()

    frames_tensor = []

    while(True):
        
        ret, frame = cap.read()

        # read over
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame = cv2.resize(frame, (224, 224))
        #frame = transform(frame).unsqueeze(0)

        frames_tensor.append(frame)

    # note: please use list to append different frame and concate them finally
    #       or else it will take muck longer time if they are concated after one frame has been just loaded
    frames_tensor = np.asarray_chkfinite(frames_tensor, dtype=np.uint8)
    # frames_tensor = kornia.image_to_tensor(frames_tensor, keepdim=False).div(255.0)
    #frames_tensor = torch.cat(frames_tensor, 0)

    return frames_tensor

def get_video_list(video_path):
    video_list = []
    for path in video_path:
        video_list.append(cv2_loadvideo(path)[None, ...])
    batched_video = np.concatenate(video_list, axis=0)
    return batched_video
    

def main():
    names, base_paths, eval_paths = generate_my_video_path_list('high')
    print(names)
    base_video = get_video_list(base_paths)
    eval_video = get_video_list(eval_paths)
    
    assert base_video.shape == eval_video.shape
    print(f"The batch video shape is {base_video.shape}")

    evaluator = fvd.cdfvd('videomae', ckpt_path=None, half_precision=True)
    score = evaluator.compute_fvd(base_video, eval_video)
    print(f"FVD score: {score}")
    

if __name__ == "__main__":
    main()
