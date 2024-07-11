import argparse

import cv2
import numpy as np
import torch
from fastapi import FastAPI
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from PIL import Image
from transformers import AutoModel, AutoProcessor

app = FastAPI()
MODEL_NAME = "microsoft/xclip-base-patch16-zero-shot"
CLIP_LEN = 32
CHUNK_DURATION = 30  # duration in seconds
SMALL_CHUNK_DURATION = 10  # smaller chunk duration in seconds
THRESHOLD_PROBABILITY = 30.0  # threshold probability in percent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)


def get_video_length(file_path):
    cap = cv2.VideoCapture(file_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return length, fps


def split_video_into_chunks(file_path, chunk_duration, fps):
    chunks = []
    cap = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    chunk_size = int(chunk_duration * fps)
    for start in range(0, total_frames, chunk_size):
        chunk_frames = []
        for i in range(start, min(start + chunk_size, total_frames)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                chunk_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if chunk_frames:
            chunks.append((chunk_frames, start / fps, (start + chunk_size) / fps))
    cap.release()
    return chunks


def read_video_opencv(file_path, indices):
    frames = []
    failed_indices = []

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Error opening video file: {file_path}")
        return frames

    max_index = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    for idx in indices:
        if idx <= max_index:
            frame = get_frame_with_opened_cap(cap, idx)
            if frame is not None:
                frames.append(frame)
            else:
                failed_indices.append(idx)
        else:
            failed_indices.append(idx)
    cap.release()

    if failed_indices:
        print(f"Failed to extract frames at indices: {failed_indices}")
    return frames


def get_frame_with_opened_cap(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = cap.read()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def sample_uniform_frame_indices(clip_len, seg_len):
    if seg_len < clip_len:
        repeat_factor = np.ceil(clip_len / seg_len).astype(int)
        indices = np.arange(seg_len).tolist() * repeat_factor
        indices = indices[:clip_len]
    else:
        spacing = seg_len // clip_len
        indices = [i * spacing for i in range(clip_len)]
    return np.array(indices).astype(np.int64)


def concatenate_frames(frames, clip_len):
    layout = {32: (4, 8)}
    rows, cols = layout[clip_len]
    combined_image = Image.new(
        "RGB", (frames[0].shape[1] * cols, frames[0].shape[0] * rows)
    )
    frame_iter = iter(frames)
    y_offset = 0
    for i in range(rows):
        x_offset = 0
        for j in range(cols):
            img = Image.fromarray(next(frame_iter))
            combined_image.paste(img, (x_offset, y_offset))
            x_offset += frames[0].shape[1]
        y_offset += frames[0].shape[0]
    return combined_image


def process_video_chunks(video_chunks, activity):
    all_results = []
    for chunk, start_time, end_time in video_chunks:
        indices = sample_uniform_frame_indices(CLIP_LEN, seg_len=len(chunk))
        video = [chunk[i] for i in indices]
        concatenated_image = concatenate_frames(video, CLIP_LEN)

        activities_list = [activity, "other"]
        inputs = processor(
            text=activities_list,
            videos=[video],  # Wrapping the list of frames in another list
            return_tensors="pt",
            padding=True,  # Activate padding
            truncation=True,  # Activate truncation
        )

        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits_per_video = outputs.logits_per_video
        probs = logits_per_video.softmax(dim=1)

        results_probs = []
        results_logits = []
        max_prob_index = torch.argmax(probs[0]).item()
        for i in range(len(activities_list)):
            current_activity = activities_list[i]
            prob = float(probs[0][i].cpu())
            logit = float(logits_per_video[0][i].cpu())
            results_probs.append((current_activity, f"Probability: {prob * 100:.2f}%"))
            results_logits.append((current_activity, f"Raw Score: {logit:.2f}"))

        likely_label = activities_list[max_prob_index]
        likely_probability = float(probs[0][max_prob_index].cpu()) * 100

        all_results.append(
            {
                "concatenated_image": concatenated_image,
                "results_probs": results_probs,
                "results_logits": results_logits,
                "likely_label": likely_label,
                "likely_probability": likely_probability,
                "video_chunk": chunk,
                "max_prob_activity": probs[0][0].item()
                * 100,  # Probability of the activity in the first position of activities_list
                "start_time": start_time,
                "end_time": end_time,
            }
        )
    return all_results


def model_interface(uploaded_video, activity):
    video_length, fps = get_video_length(uploaded_video)
    chunks = split_video_into_chunks(uploaded_video, CHUNK_DURATION, fps)
    all_results = process_video_chunks(chunks, activity)

    refined_results = []
    for result in all_results:
        if result["max_prob_activity"] > THRESHOLD_PROBABILITY:
            small_chunks = split_video_into_chunks(
                uploaded_video, SMALL_CHUNK_DURATION, fps
            )
            refined_results.extend(process_video_chunks(small_chunks, activity))
        else:
            refined_results.append(result)

    best_result = max(refined_results, key=lambda x: x["max_prob_activity"])

    # Only save the video segment if the likely label is not "other"
    if best_result["likely_label"] != "other":
        ffmpeg_extract_subclip(
            uploaded_video,
            best_result["start_time"],
            best_result["end_time"],
            targetname="best_segment.mp4",
        )
        print(
            f"Маргинальная активность обнаружена с вероятностью {best_result['likely_probability']:.2f}% на этом сегменте. Сегмент сохранен как 'best_segment.mp4'"
        )
    else:
        print("Маргинальная активность не обнаружена.")

    return best_result


@app.get("/main2")
async def main2(video_path: str):
    # Test the function directly in Jupyter/Colab
    # video_path = "/content/Shoplifting037_x264.mp4"  # Replace with your video path
    activity = (
        "robbery, stealing, pilferage, pilfering, filching, shoplifting, thievery. "
    )
    best_result = model_interface(video_path, activity)

    # Save and print results
    best_result["concatenated_image"].save("best_concatenated_image.jpg")
    print("Результаты анализа:")
    if best_result["likely_label"] != "other":
        print(
            f"Маргинальная активность обнаружена с вероятностью {best_result['likely_probability']:.2f}% на этом сегменте."
        )
    else:
        print("Маргинальная активность не обнаружена.")
