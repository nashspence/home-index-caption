# region "debugpy"


import os

if str(os.environ.get("DEBUG", "False")) == "True":
    import debugpy

    debugpy.listen(("0.0.0.0", 5678))

    if str(os.environ.get("WAIT_FOR_DEBUGPY_CLIENT", "False")) == "True":
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()
        print("Debugger attached.")
        debugpy.breakpoint()


# endregion
# region "import"


import json
import logging
import torch
import gc
import io
import ffmpeg
import subprocess
import mimetypes

from PIL import Image as PILImage
from wand.image import Image as WandImage
from home_index_module import run_server


# endregion
# region "config"


VERSION = 1
NAME = os.environ.get("NAME", "caption")

RESIZE_MAX_DIMENSION = int(os.environ.get("RESIZE_MAX_DIMENSION", 640))
VIDEO_NUMBER_OF_FRAMES = int(os.environ.get("VIDEO_NUMBER_OF_FRAMES", 20))
DEVICE = str(os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
os.environ["HF_HOME"] = str(os.environ.get("HF_HOME", "/huggingface"))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = str(
    os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
)


# endregion
# region "read images"


def resize_image_maintain_aspect(img):
    width, height = img.width, img.height
    largest_side = max(width, height)
    if largest_side > RESIZE_MAX_DIMENSION:
        ratio = RESIZE_MAX_DIMENSION / float(largest_side)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        return img.resize((new_width, new_height))
    return img


def read_video(video_path):
    probe_info = ffmpeg.probe(video_path)
    duration = float(probe_info["format"]["duration"])
    epsilon = 0.01
    safe_duration = max(0, duration - epsilon)

    timestamps = [
        i * safe_duration / (VIDEO_NUMBER_OF_FRAMES - 1)
        for i in range(VIDEO_NUMBER_OF_FRAMES)
    ]
    frames_with_timestamps = []

    def get_frame_at_time(t: float):
        out, _ = (
            ffmpeg.input(video_path, ss=t)
            .output("pipe:", vframes=1, format="image2", vcodec="png")
            .run(capture_stdout=True, capture_stderr=True)
        )
        return out

    for i, t in enumerate(timestamps):
        out = get_frame_at_time(t)
        if i == VIDEO_NUMBER_OF_FRAMES - 1 and len(out) == 0:
            fallback_time = t
            step_back = 0.05
            retries = 10
            while len(out) == 0 and fallback_time > 0 and retries > 0:
                fallback_time -= step_back
                out = get_frame_at_time(fallback_time)
                retries -= 1
        if len(out) > 0:
            image = PILImage.open(io.BytesIO(out)).convert("RGB")
            resized_image = resize_image_maintain_aspect(image)
            frames_with_timestamps.append((resized_image, t))
        else:
            frames_with_timestamps.append((None, t))

    return frames_with_timestamps


def read_image(file_path):
    with WandImage(filename=file_path, resolution=300) as img:
        img.auto_orient()
        img.format = "png"
        first_frame = img.sequence[0]
        with WandImage(image=first_frame, resolution=300) as single_frame:
            img.auto_orient()
            single_frame.format = "png"
            blob = single_frame.make_blob()
            pillow_image = PILImage.open(io.BytesIO(blob)).convert("RGB")
            resized_image = resize_image_maintain_aspect(pillow_image)
            return (resized_image, 0)


# endregion
# region "hello"


def hello():
    return {
        "name": NAME,
        "version": VERSION,
        "filterable_attributes": [f"{NAME}.text"],
        "sortable_attributes": [],
    }


# endregion
# region "load/unload"


model = None
processor = None
summarizer = None


def load():
    global model, processor, summarizer
    from transformers import BlipProcessor, BlipForConditionalGeneration

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    )
    model.to(DEVICE)


def unload():
    global model, processor

    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()


# endregion
# region "check/run"


def get_supported_formats():
    result = subprocess.run(
        ["identify", "-list", "format"], capture_output=True, text=True
    )
    lines = result.stdout.splitlines()
    supported = set()
    for line in lines:
        line = line.strip()
        if line and not line.startswith("-") and not line.startswith("Format"):
            parts = line.split()
            if len(parts) >= 2:
                ext = parts[0].lower().rstrip("*")
                modes = parts[2]
                if "r" in modes:
                    supported.add(ext)
    return supported


SUPPORTED_FORMATS = get_supported_formats()


def get_extensions_from_mime(mime_type):
    return mimetypes.guess_all_extensions(mime_type)


def can_wand_open(mime_type):
    extensions = get_extensions_from_mime(mime_type)
    for ext in extensions:
        if ext.lstrip(".").lower() in SUPPORTED_FORMATS:
            return True
    return False


def check(file_path, document, metadata_dir_path):
    version_path = metadata_dir_path / "version.json"
    version = None

    if version_path.exists():
        with open(version_path, "r") as file:
            version = json.load(file)

    if version and version.get("version") == VERSION:
        return False

    if document["type"].startswith("audio/"):
        return False

    return can_wand_open(document["type"])


def run(file_path, document, metadata_dir_path):
    global reader
    logging.info(f"start {file_path}")

    version_path = metadata_dir_path / "version.json"
    frame_captions_path = metadata_dir_path / "frame_captions.json"

    exception = None
    try:
        frame_captions = []
        if document["type"].startswith("video/"):
            frames = read_video(file_path)
        else:
            frames = [read_image(file_path)]
        for image, timestamp in frames:
            inputs = processor(images=image, return_tensors="pt").to(DEVICE)
            outputs = model.generate(**inputs)
            caption = processor.decode(outputs[0], skip_special_tokens=True)
            frame_captions.append((timestamp, caption))
        caption = " ".join([caption for _, caption in frame_captions])
        document[NAME] = {}
        document[NAME]["text"] = caption
        with open(frame_captions_path, "w") as file:
            json.dump(frame_captions, file, indent=4)
    except FileNotFoundError as e:
        raise e
    except Exception as e:
        exception = e
        logging.exception("failed")

    with open(version_path, "w") as file:
        json.dump({"version": VERSION, "exception": str(exception)}, file, indent=4)

    logging.info("done")
    return document


# endregion

if __name__ == "__main__":
    run_server(NAME, hello, check, run, load, unload)
