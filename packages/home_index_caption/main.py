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
    frames = []

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
            frames.append(resized_image)
        else:
            frames.append(None)
    return frames


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
            return resized_image


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


def load():
    global model, processor
    print(
        f"cuda_mem alloc={torch.cuda.memory_allocated() / (1024 ** 3):.2f}GB reserved={torch.cuda.memory_reserved() / (1024 ** 3):.2f}GB"
    )

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
    print(
        f"cuda_mem alloc={torch.cuda.memory_allocated() / (1024 ** 3):.2f}GB reserved={torch.cuda.memory_reserved() / (1024 ** 3):.2f}GB"
    )


# endregion
# region "check/run"


def check(file_path, document, metadata_dir_path):
    version_path = metadata_dir_path / "version.json"
    version = None
    if version_path.exists():
        with open(version_path, "r") as file:
            version = json.load(file)

    if version and version["version"] == VERSION:
        return False
    if document["type"].startswith("audio/"):
        return False
    if document["type"].startswith("video/"):
        return True
    try:
        with WandImage(filename=file_path):
            return True
    except Exception:
        return False


def run(file_path, document, metadata_dir_path):
    global reader
    logging.info(f"start {file_path}")

    version_path = metadata_dir_path / "version.json"
    plaintext_path = metadata_dir_path / "caption.txt"

    exception = None
    try:
        captions = []
        if document["type"].startswith("video/"):
            images = read_video(file_path)
        else:
            images = [read_image(file_path)]
        for image in images:
            inputs = processor(images=image, return_tensors="pt").to(DEVICE)
            outputs = model.generate(**inputs)
            caption = processor.decode(outputs[0], skip_special_tokens=True)
            captions.append(caption)
        caption = " ".join(captions)
        document[NAME] = {}
        document[NAME]["text"] = caption
        with open(plaintext_path, "w") as file:
            file.write(caption)
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
