#!/usr/bin/env python3
"""
Download all model weights and datasets used in the multimodal course notebooks.
This script pre-downloads all HuggingFace models and datasets to populate the cache.
"""

import subprocess
import torch
from datasets import load_dataset
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


def download_dataset_meld_audio():
    print("Downloading ajyy/MELD_audio dataset (train split)...")
    load_dataset("ajyy/MELD_audio", split="train", trust_remote_code=True)


def download_dataset_librispeech_dummy():
    print("Downloading hf-internal-testing/librispeech_asr_dummy dataset (validation split)...")
    load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")


def download_dataset_librispeech_long():
    print("Downloading distil-whisper/librispeech_long dataset (validation split)...")
    load_dataset("distil-whisper/librispeech_long", "clean", split="validation")


def download_dataset_coco():
    print("Downloading detection-datasets/coco dataset (val split, streaming - 100 samples)...")
    dataset = load_dataset("detection-datasets/coco", split="val", streaming=True)
    # Download 100 samples to populate cache
    for i, _ in enumerate(dataset):
        if i >= 99:  # 0-99 = 100 samples
            break


def download_dataset_oxford_pets():
    print("Downloading enterprise-explorers/oxford-pets dataset (train split)...")
    load_dataset("enterprise-explorers/oxford-pets", split="train")


def download_dataset_esc50():
    print("Downloading ashraq/esc50 dataset (train split)...")
    load_dataset("ashraq/esc50", split="train")


def download_dataset_magicbrush():
    print("Downloading osunlp/MagicBrush dataset (dev split, streaming - 1000 samples)...")
    dataset = load_dataset("osunlp/MagicBrush", split="dev", streaming=True)
    # Download 1000 samples to populate cache
    for i, _ in enumerate(dataset):
        if i >= 999:  # 0-999 = 1000 samples
            break


def download_dataset_food101():
    print("Downloading food101 dataset (validation split)...")
    load_dataset("food101", split="validation")


def download_dataset_realworldqa():
    print("Downloading xai-org/RealworldQA dataset (test split)...")
    load_dataset("xai-org/RealworldQA", split="test")


def main():
    print("Starting model weight and dataset downloads...")
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
        download_dataset_meld_audio,
        download_dataset_librispeech_dummy,
        download_dataset_librispeech_long,
        download_dataset_coco,
        download_dataset_oxford_pets,
        download_dataset_esc50,
        download_dataset_magicbrush,
        download_dataset_food101,
        download_dataset_realworldqa,
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
    print("\nSummary:")
    print("  - Downloaded 13 HuggingFace model weights")
    print("  - Downloaded 9 HuggingFace datasets")
    print("  - Downloaded 1 YOLOv9 model (if yolo package installed)")
    print("  - Downloaded 1 Ollama model (if ollama CLI installed)")
    print("\nNotes:")
    print("  - YOLOv9: Requires 'yolo' package. If not installed, this model was skipped.")
    print("  - Ollama: Requires 'ollama' CLI. If not installed, run 'ollama pull qwen2.5vl:3b' manually.")
    print("  - Streaming datasets: COCO (100 samples), MagicBrush (1000 samples)")
    print("  - Full datasets: librispeech, oxford-pets, esc50, food101, RealworldQA, MELD_audio")


if __name__ == "__main__":
    main()

