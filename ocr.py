"""
ocr.py
Capture a still image from PiCamera and run OCR via OpenRouter.

Usage:
  python ocr.py
  python ocr.py --width 1280 --height 720
  python ocr.py --config ocr_config.json
"""

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

import cv2

try:
    from picamera2 import Picamera2
    PICAMERA2_ERR = None
except Exception as exc:  # pragma: no cover
    Picamera2 = None
    PICAMERA2_ERR = exc

try:
    import picamera
    PICAMERA_ERR = None
except Exception as exc:  # pragma: no cover
    picamera = None
    PICAMERA_ERR = exc


DEFAULT_CONFIG_PATH = "ocr_config.json"
DEFAULT_CAPTURE_DIR = "ocr_captures"
DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


class ConfigError(Exception):
    pass


def _load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise ConfigError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    api_key = data.get("key") or data.get("openrouter_api_key") or data.get("api_key")
    prompt = data.get("prompt")
    if not api_key:
        raise ConfigError("Missing 'key' in ocr_config.json")
    if not prompt:
        raise ConfigError("Missing 'prompt' in ocr_config.json")

    return {
        "api_key": api_key,
        "prompt": prompt,
        "model": data.get("model", DEFAULT_MODEL),
        "base_url": data.get("base_url", DEFAULT_BASE_URL),
        "timeout_s": data.get("timeout_s", 60),
        "max_tokens": data.get("max_tokens", 512),
        "temperature": data.get("temperature", 0.0),
        "app_name": data.get("app_name"),
        "app_url": data.get("app_url"),
    }


def _capture_image(width: int, height: int, warmup_s: float, output_path: str) -> bytes:
    if Picamera2 is not None:
        picam2 = Picamera2()
        config = picam2.create_still_configuration(main={"size": (width, height)})
        picam2.configure(config)
        picam2.start()
        time.sleep(warmup_s)

        frame = picam2.capture_array()
        picam2.stop()

        if frame is None:
            raise RuntimeError("Failed to capture image from PiCamera (picamera2)")

        # picamera2 returns RGB; OpenCV expects BGR
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not cv2.imwrite(output_path, frame):
            raise RuntimeError(f"Failed to write image: {output_path}")

        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("Failed to encode image to JPEG")
        return buf.tobytes()

    if picamera is not None:
        import io

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        stream = io.BytesIO()
        with picamera.PiCamera() as camera:
            camera.resolution = (width, height)
            time.sleep(warmup_s)
            camera.capture(stream, format="jpeg")
        image_bytes = stream.getvalue()
        if not image_bytes:
            raise RuntimeError("Failed to capture image from PiCamera (picamera)")

        with open(output_path, "wb") as f:
            f.write(image_bytes)
        return image_bytes

    # Fallback: use rpicam-still / libcamera-still CLI if available (Ubuntu often lacks picamera2)
    for cmd in ("rpicam-still", "libcamera-still"):
        if shutil.which(cmd):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            timeout_ms = max(1000, int(warmup_s * 1000))
            result = subprocess.run(
                [
                    cmd,
                    "--nopreview",
                    "--timeout",
                    str(timeout_ms),
                    "--width",
                    str(width),
                    "--height",
                    str(height),
                    "-o",
                    output_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="ignore").strip()
                raise RuntimeError(f"{cmd} failed: {stderr}")
            with open(output_path, "rb") as f:
                return f.read()

    raise RuntimeError(
        "No PiCamera library or CLI available. Install picamera2, picamera, "
        "or rpicam-apps/libcamera-apps. "
        f"picamera2 import error: {PICAMERA2_ERR}; picamera import error: {PICAMERA_ERR}"
    )


def _call_openrouter(cfg: dict, image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    payload = {
        "model": cfg["model"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": cfg["prompt"]},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": cfg["max_tokens"],
        "temperature": cfg["temperature"],
    }

    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }
    if cfg.get("app_name"):
        headers["X-Title"] = str(cfg["app_name"])
    if cfg.get("app_url"):
        headers["HTTP-Referer"] = str(cfg["app_url"])

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(cfg["base_url"], data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=cfg["timeout_s"]) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {err_body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"OpenRouter request failed: {e}") from e

    parsed = json.loads(body)
    try:
        return parsed["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        raise RuntimeError(f"Unexpected OpenRouter response: {parsed}") from exc


def _default_output_path() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(DEFAULT_CAPTURE_DIR, f"ocr_{ts}.jpg")


def main() -> int:
    parser = argparse.ArgumentParser(description="PiCamera OCR via OpenRouter")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--warmup", type=float, default=0.8)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    try:
        cfg = _load_config(args.config)
        output_path = args.output or _default_output_path()
        image_bytes = _capture_image(args.width, args.height, args.warmup, output_path)
        text = _call_openrouter(cfg, image_bytes)
        print(text)
        print(f"\n[image saved] {output_path}")
        return 0
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
