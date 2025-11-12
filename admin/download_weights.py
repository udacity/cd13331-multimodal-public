#!/usr/bin/env python3
"""
Download all model weights used in the multimodal course notebooks.
This script pre-downloads all HuggingFace models to populate the cache.
"""

import subprocess
import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    CLIPProcessor,
    CLIPModel,
    CLIPVisionModelWithProjection,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageClassification,
    RTDetrImageProcessor,
    RTDetrV2ForObjectDetection,
    Sam2Processor,
    Sam2Model,
    ClapModel,
    ClapProcessor,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoModelForSpeechSeq2Seq,
    pipeline,
)

try:
    from hydra import compose, initialize_config_module
    from yolo.tools.solver import InferenceModel
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


def download_smolvlm():
    print("Downloading HuggingFaceTB/SmolVLM2-500M-Video-Instruct...")
    AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
    )


def download_clip_base():
    print("Downloading openai/clip-vit-base-patch32...")
    CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", fast_processor=True)
    CLIPModel.from_pretrained("openai/clip-vit-base-patch32")


def download_clip4clip():
    print("Downloading Searchium-ai/clip4clip-webvid150k...")
    CLIPVisionModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k")
    CLIPTextModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k")
    CLIPTokenizer.from_pretrained("Searchium-ai/clip4clip-webvid150k")


def download_vit():
    print("Downloading google/vit-base-patch16-224...")
    AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")


def download_siglip():
    print("Downloading google/siglip-base-patch16-224...")
    AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224")


def download_dinov2_registers():
    print("Downloading facebook/dinov2-with-registers-base...")
    AutoImageProcessor.from_pretrained("facebook/dinov2-with-registers-base")
    AutoModel.from_pretrained("facebook/dinov2-with-registers-base")


def download_dinov2_base():
    print("Downloading facebook/dinov2-base...")
    AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    AutoModelForImageClassification.from_pretrained(
        "facebook/dinov2-base",
        num_labels=2,
        id2label={0: "class_0", 1: "class_1"},
        label2id={"class_0": 0, "class_1": 1},
        ignore_mismatched_sizes=True,
    )


def download_rtdetr():
    print("Downloading PekingU/rtdetr_v2_r50vd...")
    RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r50vd")
    RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r50vd")


def download_rtdetr_coco():
    print("Downloading PekingU/rtdetr_r50vd_coco_o365...")
    RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
    RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")


def download_sam2():
    print("Downloading facebook/sam2.1-hiera-base-plus...")
    Sam2Processor.from_pretrained("facebook/sam2.1-hiera-base-plus")
    Sam2Model.from_pretrained("facebook/sam2.1-hiera-base-plus")


def download_florence():
    print("Downloading ducviet00/Florence-2-base-hf...")
    device = 0 if torch.cuda.is_available() else -1
    pipeline(
        "image-text-to-text",
        model="ducviet00/Florence-2-base-hf",
        device=device,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )


def download_clap():
    print("Downloading laion/clap-htsat-unfused...")
    ClapModel.from_pretrained("laion/clap-htsat-unfused")
    ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    device = 0 if torch.cuda.is_available() else -1
    pipeline("zero-shot-audio-classification", model="laion/clap-htsat-unfused", device=device)


def download_whisper():
    print("Downloading openai/whisper-large-v3-turbo...")
    WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-large-v3-turbo",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=torch_dtype,
        device=device,
    )


def download_yolov9():
    if not YOLO_AVAILABLE:
        raise ImportError("YOLOv9 package not installed. Install with 'pip install yolo' or skip this model.")

    print("Downloading YOLOv9 v9-s weights...")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    torch.set_default_device(device)

    with initialize_config_module(config_module="yolo.config", version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "task.task=inference",
                "model=v9-s",
                "use_wandb=false",
                f"device={device}",
            ],
        )

    model = InferenceModel(cfg).to(device)
    model.eval()
    model.setup(cfg.task.task)


def download_ollama_qwen():
    print("Downloading qwen2.5vl:3b via Ollama...")
    result = subprocess.run(
        ["ollama", "pull", "qwen2.5vl:3b"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Ollama command failed: {result.stderr}")
    print(result.stdout)


def main():
    print("Starting model weight downloads...")
    print("=" * 60)

    models = [
        download_smolvlm,
        download_clip_base,
        download_clip4clip,
        download_vit,
        download_siglip,
        download_dinov2_registers,
        download_dinov2_base,
        download_rtdetr,
        download_rtdetr_coco,
        download_sam2,
        download_florence,
        download_clap,
        download_whisper,
        download_yolov9,
        download_ollama_qwen,
    ]

    for i, model_func in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}]")
        try:
            model_func()
            print("✓ Success")
        except Exception as e:
            print(f"✗ Error: {e}")
            print("Continuing with next model...")

    print("\n" + "=" * 60)
    print("Download complete!")
    print("\nNotes:")
    print("  - YOLOv9: Requires 'yolo' package. If not installed, this model was skipped.")
    print("  - Ollama: Requires 'ollama' CLI. If not installed, run 'ollama pull qwen2.5vl:3b' manually.")


if __name__ == "__main__":
    main()
