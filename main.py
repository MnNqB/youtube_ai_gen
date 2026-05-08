import os
import tempfile
import subprocess
import glob
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


class AnalyzeRequest(BaseModel):
    url: str


def extract_frames(video_path: str, num_frames: int = 15) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError("Could not read video frames")

    indices = np.linspace(0, total - 1, num=min(num_frames, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def compute_noise_stats(frames: list[np.ndarray]) -> tuple[float, float]:
    noise_stds = []
    noise_means = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        abs_lap = np.abs(laplacian)
        noise_stds.append(float(np.std(abs_lap)))
        noise_means.append(float(np.mean(abs_lap)))
    return float(np.mean(noise_stds)), float(np.mean(noise_means))


def classify(noise_std: float, noise_mean: float) -> dict:
    is_sora = noise_std < 4.5 and noise_mean < 2.0
    is_veo = noise_std > 6.5 and noise_mean > 3.0

    if is_sora:
        label = "AI-Generated (Sora-style)"
        confidence = min(100, int(((4.5 - noise_std) / 4.5 + (2.0 - noise_mean) / 2.0) / 2 * 100 + 70))
        detail = "Noise signature is unusually clean — consistent with aggressive denoising by diffusion models like Sora."
    elif is_veo:
        label = "AI-Generated (Veo-style)"
        confidence = min(100, int(((noise_std - 6.5) / 6.5 + (noise_mean - 3.0) / 3.0) / 2 * 100 + 70))
        detail = "Elevated diffusion noise detected — characteristic of Veo and similar high-noise generative models."
    else:
        # Score how close to the real range (noise_std ~4.5–6.5, noise_mean ~2.0–3.0)
        std_real_center = 5.5
        mean_real_center = 2.5
        std_dist = abs(noise_std - std_real_center) / std_real_center
        mean_dist = abs(noise_mean - mean_real_center) / mean_real_center
        distance = (std_dist + mean_dist) / 2
        confidence = max(50, min(95, int((1 - distance) * 95)))
        label = "Likely Real"
        detail = "Noise characteristics fall within the expected range for real video — normal compression artifacts present."

    return {
        "label": label,
        "confidence": confidence,
        "noise_std": round(noise_std, 2),
        "noise_mean": round(noise_mean, 2),
        "detail": detail,
    }


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_template = os.path.join(tmpdir, "video.%(ext)s")
        result = subprocess.run(
            [
                "yt-dlp",
                "--no-playlist",
                "--extractor-args", "youtube:player_client=android,ios,web",
                "-f", "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio/best[height<=720]/best",
                "--merge-output-format", "mp4",
                "-o", out_template,
                url,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise HTTPException(status_code=400, detail=f"Could not download video: {result.stderr[-300:]}")

        matches = glob.glob(os.path.join(tmpdir, "video.*"))
        if not matches:
            raise HTTPException(status_code=500, detail="Downloaded file not found")

        video_path = matches[0]
        try:
            frames = extract_frames(video_path)
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))

        if not frames:
            raise HTTPException(status_code=500, detail="No frames could be extracted")

        noise_std, noise_mean = compute_noise_stats(frames)
        return classify(noise_std, noise_mean)
