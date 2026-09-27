"""Synthetic degradations matching the robustness requirements of the task."""
import cv2
import numpy as np


def skew(img, deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    border = tuple(int(v) for v in np.median(img[:10, :10].reshape(-1, 3), axis=0))
    return cv2.warpAffine(img, M, (w, h), borderValue=border)


def uneven_light(img, strength=0.45):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = 1.0 - strength * ((xx / w) * 0.7 + (yy / h) * 0.3)
    return np.clip(img.astype(np.float32) * g[..., None], 0, 255).astype(np.uint8)


def stains(img, n=5, seed=1):
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32)
    h, w = img.shape[:2]
    for _ in range(n):
        m = np.zeros((h, w), np.float32)
        cv2.circle(m, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(40, 160)), 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), 25)[..., None]
        out = out * (1 - 0.35 * m) + np.array([40, 90, 140], np.float32) * 0.35 * m
    return np.clip(out, 0, 255).astype(np.uint8)


def blur_noise(img, sigma=1.4, noise=9, seed=2):
    rng = np.random.default_rng(seed)
    out = cv2.GaussianBlur(img, (0, 0), sigma).astype(np.float32)
    out += rng.normal(0, noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def fade(img, amount=0.45):
    """Faded ink: pull everything toward the paper colour."""
    bg = np.percentile(img.reshape(-1, 3), 90, axis=0)
    return np.clip(img * (1 - amount) + bg * amount, 0, 255).astype(np.uint8)


def all_degradations(img):
    return blur_noise(stains(uneven_light(skew(fade(img, 0.35), 2.5))))
