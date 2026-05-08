import os
import base64
import glob
import subprocess
import tempfile
import uuid
from io import BytesIO

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request

load_dotenv()
COOKIES_PATH = os.getenv("COOKIES_PATH", "cookies.txt")

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

VALID_MAGIC = [
    (4, 8, b"ftyp"),    # MP4 / MOV
    (0, 4, b"\x1a\x45\xdf\xa3"),  # WebM / MKV
    (0, 4, b"RIFF"),    # AVI (also check offset 8)
]


def _is_valid_video_bytes(header: bytes) -> bool:
    if header[4:8] == b"ftyp":
        return True
    if header[0:4] == b"\x1a\x45\xdf\xa3":
        return True
    if header[0:4] == b"RIFF" and header[8:12] == b"AVI ":
        return True
    return False


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
    noise_stds, noise_means = [], []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        abs_lap = np.abs(laplacian)
        noise_stds.append(float(np.std(abs_lap)))
        noise_means.append(float(np.mean(abs_lap)))
    return float(np.mean(noise_stds)), float(np.mean(noise_means))


def generate_heatmap_b64(frame: np.ndarray) -> str:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    norm = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    _, buf = cv2.imencode(".png", heatmap)
    return base64.b64encode(buf.tobytes()).decode()


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


def _run_analysis(video_path: str) -> dict:
    frames = extract_frames(video_path)
    if not frames:
        raise HTTPException(status_code=500, detail="No frames could be extracted")
    noise_std, noise_mean = compute_noise_stats(frames)
    result = classify(noise_std, noise_mean)
    mid = frames[len(frames) // 2]
    result["heatmap"] = generate_heatmap_b64(mid)
    return result


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze/upload")
async def analyze_upload(file: UploadFile = File(...)):
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 50 MB limit")
    if len(data) < 12 or not _is_valid_video_bytes(data[:12]):
        raise HTTPException(status_code=400, detail="File does not appear to be a valid video (MP4, MOV, WebM, MKV, AVI)")

    tmp_path = f"/tmp/{uuid.uuid4().hex}.mp4"
    try:
        with open(tmp_path, "wb") as f:
            f.write(data)
        return _run_analysis(tmp_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


class UrlRequest(BaseModel):
    url: str
    cookies: str = ""


@app.post("/analyze/url")
def analyze_url(req: UrlRequest):
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    cookie_tmp: str | None = None
    yt_cmd_base = [
        "yt-dlp",
        "--no-playlist",
        "--extractor-args", "youtube:player_client=android,ios,web",
        "-f", "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "--merge-output-format", "mp4",
    ]

    try:
        # Determine which cookie source to use
        cookies_text = req.cookies.strip()
        if cookies_text:
            cookie_tmp = f"/tmp/{uuid.uuid4().hex}.txt"
            with open(cookie_tmp, "w") as f:
                f.write(cookies_text)
            os.chmod(cookie_tmp, 0o600)
            yt_cmd_base += ["--cookies", cookie_tmp]
        elif os.path.isfile(COOKIES_PATH):
            yt_cmd_base += ["--cookies", COOKIES_PATH]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_template = os.path.join(tmpdir, "video.%(ext)s")
            cmd = yt_cmd_base + ["-o", out_template, url]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                stderr = result.stderr
                needs_cookies = any(kw in stderr for kw in [
                    "Sign in", "bot", "cookies", "authentication", "age-restricted",
                    "This video is unavailable", "Private video",
                ])
                if needs_cookies:
                    raise HTTPException(
                        status_code=403,
                        detail={"error_type": "cookies_required", "message": "YouTube requires authentication for this video. Please paste your cookies below."},
                    )
                raise HTTPException(status_code=400, detail=f"Could not download video: {stderr[-300:]}")

            matches = glob.glob(os.path.join(tmpdir, "video.*"))
            if not matches:
                raise HTTPException(status_code=500, detail="Downloaded file not found")

            return _run_analysis(matches[0])

    except HTTPException:
        raise
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Download timed out — video may be too long")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cookie_tmp and os.path.exists(cookie_tmp):
            os.remove(cookie_tmp)
