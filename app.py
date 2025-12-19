import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison

# ============================================================
# Robust path resolution (works whether app.py is in repo root or subfolder)
# ============================================================
APP_DIR = Path(__file__).resolve().parent
CWD_DIR = Path.cwd().resolve()
PARENT_DIR = APP_DIR.parent.resolve()


def _pick_dir(folder_name: str, must_contain_ext: Optional[str] = None) -> Path:
    """
    Pick a directory among common candidates:
      - APP_DIR/folder_name
      - CWD_DIR/folder_name   (repo root on Streamlit Cloud)
      - PARENT_DIR/folder_name
    If must_contain_ext is provided, prefer the first directory that contains files of that ext.
    Otherwise prefer the first existing directory.
    If none exist, create APP_DIR/folder_name.
    """
    candidates = [
        APP_DIR / folder_name,
        CWD_DIR / folder_name,
        PARENT_DIR / folder_name,
    ]

    if must_contain_ext:
        for d in candidates:
            if d.exists() and d.is_dir() and any(d.glob(f"*{must_contain_ext}")):
                return d

    for d in candidates:
        if d.exists() and d.is_dir():
            return d

    d = APP_DIR / folder_name
    d.mkdir(parents=True, exist_ok=True)
    return d


LUT_DIR = _pick_dir("luts", must_contain_ext=".cube")
ASSETS_DIR = _pick_dir("assets")
PRESETS_DIR = _pick_dir("presets")


def _find_example_image() -> Optional[Path]:
    candidates = [
        ASSETS_DIR / "example.png",
        ASSETS_DIR / "example.jpg",
        ASSETS_DIR / "example.jpeg",
        APP_DIR / "example.png",
        CWD_DIR / "assets" / "example.png",
        PARENT_DIR / "assets" / "example.png",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


EXAMPLE_IMG_PATH = _find_example_image()
PRESETS_PATH = PRESETS_DIR / "scene_presets.json"

# Export backend config
PRESERVE_AUDIO_ALWAYS_ON = True
ENABLE_LOCAL_SAVE_DIALOG = True  # local-only "Save As..." dialog via tkinter

st.set_page_config(page_title="Video LUT Previewer", layout="wide")
st.title("🎞️ Video LUT Previewer (.cube)")
st.caption("Upload a video, preview 3 frames with a before/after slider, compare LUTs, then export.")

# -----------------------------
# Local GUI helpers
# -----------------------------
def running_on_streamlit_cloud() -> bool:
    return bool(os.environ.get("STREAMLIT_CLOUD")) or "streamlit.app" in os.environ.get("HOSTNAME", "")


def gui_available() -> bool:
    if os.name == "nt":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


# -----------------------------
# LUT parsing + application
# -----------------------------
@dataclass
class CubeLUT:
    size: int
    domain_min: np.ndarray
    domain_max: np.ndarray
    table: np.ndarray  # [b,g,r,3]


CUBE_RE_FLOAT = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")


def _parse_floats(line: str) -> List[float]:
    return [float(x) for x in CUBE_RE_FLOAT.findall(line)]


@st.cache_data(show_spinner=False)
def load_cube_lut(cube_bytes: bytes, assume_bgr_major: bool = True) -> CubeLUT:
    text = cube_bytes.decode("utf-8", errors="ignore").splitlines()

    size = None
    domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    data = []
    for raw in text:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        u = line.upper()
        if u.startswith("LUT_3D_SIZE"):
            size = int(line.split()[-1])
            continue
        if u.startswith("DOMAIN_MIN"):
            vals = _parse_floats(line)
            if len(vals) >= 3:
                domain_min = np.array(vals[:3], dtype=np.float32)
            continue
        if u.startswith("DOMAIN_MAX"):
            vals = _parse_floats(line)
            if len(vals) >= 3:
                domain_max = np.array(vals[:3], dtype=np.float32)
            continue

        vals = _parse_floats(line)
        if len(vals) == 3:
            data.append(vals)

    if size is None:
        raise ValueError("Could not find LUT_3D_SIZE in .cube file.")

    expected = size * size * size
    if len(data) < expected:
        raise ValueError(f".cube incomplete: got {len(data)} rows, expected {expected}.")

    data = np.asarray(data[:expected], dtype=np.float32)
    table = data.reshape((size, size, size, 3))

    if not assume_bgr_major:
        table = table.transpose(2, 1, 0, 3)

    return CubeLUT(size=size, domain_min=domain_min, domain_max=domain_max, table=table)


def apply_lut_trilinear(rgb_u8: np.ndarray, lut: CubeLUT, strength: float = 1.0) -> np.ndarray:
    if rgb_u8.dtype != np.uint8:
        raise ValueError("apply_lut_trilinear expects uint8 RGB input.")

    img = rgb_u8.astype(np.float32) / 255.0

    dom_min = lut.domain_min.reshape((1, 1, 3))
    dom_max = lut.domain_max.reshape((1, 1, 3))
    denom = np.maximum(dom_max - dom_min, 1e-8)

    x = (img - dom_min) / denom
    x = np.clip(x, 0.0, 1.0)

    s = lut.size
    x = x * (s - 1)

    r = x[..., 0]
    g = x[..., 1]
    b = x[..., 2]

    r0 = np.floor(r).astype(np.int32)
    g0 = np.floor(g).astype(np.int32)
    b0 = np.floor(b).astype(np.int32)

    r1 = np.clip(r0 + 1, 0, s - 1)
    g1 = np.clip(g0 + 1, 0, s - 1)
    b1 = np.clip(b0 + 1, 0, s - 1)

    dr = (r - r0).astype(np.float32)
    dg = (g - g0).astype(np.float32)
    db = (b - b0).astype(np.float32)

    H, W = r0.shape
    r0f = r0.reshape(-1)
    r1f = r1.reshape(-1)
    g0f = g0.reshape(-1)
    g1f = g1.reshape(-1)
    b0f = b0.reshape(-1)
    b1f = b1.reshape(-1)

    drf = dr.reshape(-1)[:, None]
    dgf = dg.reshape(-1)[:, None]
    dbf = db.reshape(-1)[:, None]

    T = lut.table

    c000 = T[b0f, g0f, r0f]
    c100 = T[b0f, g0f, r1f]
    c010 = T[b0f, g1f, r0f]
    c110 = T[b0f, g1f, r1f]
    c001 = T[b1f, g0f, r0f]
    c101 = T[b1f, g0f, r1f]
    c011 = T[b1f, g1f, r0f]
    c111 = T[b1f, g1f, r1f]

    c00 = c000 * (1 - drf) + c100 * drf
    c01 = c001 * (1 - drf) + c101 * drf
    c10 = c010 * (1 - drf) + c110 * drf
    c11 = c011 * (1 - drf) + c111 * drf

    c0 = c00 * (1 - dgf) + c10 * dgf
    c1 = c01 * (1 - dgf) + c11 * dgf

    out = c0 * (1 - dbf) + c1 * dbf
    out = out.reshape(H, W, 3)
    out = np.clip(out, 0.0, 1.0)

    if strength < 1.0:
        out = img * (1.0 - strength) + out * strength

    return (out * 255.0 + 0.5).astype(np.uint8)


# -----------------------------
# LUT library (disk + session uploads)
# -----------------------------
@st.cache_data(show_spinner=False)
def list_lut_paths() -> List[Path]:
    LUT_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(LUT_DIR.glob("*.cube"), key=lambda p: p.name.lower())


@st.cache_data(show_spinner=False)
def load_lut_bytes_map_disk() -> Dict[str, bytes]:
    return {p.name: p.read_bytes() for p in list_lut_paths()}


def get_lut_map() -> Dict[str, bytes]:
    m = load_lut_bytes_map_disk()
    m.update(st.session_state.get("uploaded_luts", {}))
    return m


# -----------------------------
# Video helpers
# -----------------------------
def _write_uploaded_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1].lower() or ".mp4"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return path


def _read_frame_at(cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(frame_idx, 0)))
    ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        return None
    return frame_bgr


@st.cache_data(show_spinner=False)
def extract_three_frames(video_path: str) -> Tuple[List[np.ndarray], Dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    if frame_count <= 0:
        frame_count = 1

    idxs = [0, frame_count // 2, max(frame_count - 1, 0)]
    frames = []
    for idx in idxs:
        fr = _read_frame_at(cap, idx)
        if fr is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, fr = cap.read()
            if not ok:
                fr = np.zeros((720, 1280, 3), dtype=np.uint8)
        frames.append(fr)

    cap.release()
    meta = {"frame_count": frame_count, "fps": fps, "idxs": idxs}
    return frames, meta


def bgr_to_rgb_u8(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


# -----------------------------
# Temp image saving for image_comparison
# -----------------------------
def get_session_tmpdir() -> str:
    if "tmpdir" not in st.session_state:
        st.session_state["tmpdir"] = tempfile.mkdtemp(prefix="kslut_")
    return st.session_state["tmpdir"]


def save_rgb_to_png_path(rgb_u8: np.ndarray, name: str) -> str:
    tmpdir = get_session_tmpdir()
    path = os.path.join(tmpdir, f"{name}.png")
    Image.fromarray(rgb_u8, mode="RGB").save(path, format="PNG")
    return path


# -----------------------------
# Presets
# -----------------------------
def load_presets() -> List[dict]:
    if not PRESETS_PATH.exists():
        return []
    try:
        return json.loads(PRESETS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_preset(entry: dict) -> None:
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    presets = load_presets()
    presets.append(entry)
    PRESETS_PATH.write_text(json.dumps(presets, indent=2), encoding="utf-8")


def queue_apply_preset(p: dict) -> None:
    st.session_state["_pending_preset"] = p
    st.rerun()


def apply_preset_early(p: dict, lut_names: List[str]) -> None:
    st.session_state["compare_mode"] = p.get("compare_mode", "LUT A vs Original")
    st.session_state["assume_order"] = bool(p.get("assume_order", False))
    st.session_state["strength_a"] = float(p.get("strength_a", 1.0))

    lut_a = p.get("lut_a_name", lut_names[0] if lut_names else None)
    st.session_state["lut_a_name"] = lut_a if lut_a in lut_names else (lut_names[0] if lut_names else None)

    if st.session_state["compare_mode"] == "LUT A vs LUT B":
        st.session_state["strength_b"] = float(p.get("strength_b", 1.0))
        default_b = lut_names[min(1, len(lut_names) - 1)] if lut_names else None
        lut_b = p.get("lut_b_name", default_b)
        st.session_state["lut_b_name"] = lut_b if lut_b in lut_names else default_b


# ============================================================
# Session defaults
# ============================================================
st.session_state.setdefault("uploaded_luts", {})
st.session_state.setdefault("assume_order", False)
st.session_state.setdefault("compare_mode", "LUT A vs Original")
st.session_state.setdefault("strength_a", 1.0)
st.session_state.setdefault("strength_b", 1.0)

lut_map = get_lut_map()
lut_names = sorted(lut_map.keys())

if "_pending_preset" in st.session_state:
    pending = st.session_state.pop("_pending_preset")
    lut_map = get_lut_map()
    lut_names = sorted(lut_map.keys())
    if lut_names:
        apply_preset_early(pending, lut_names)

if lut_names:
    st.session_state.setdefault("lut_a_name", lut_names[0])
    st.session_state.setdefault("lut_b_name", lut_names[min(1, len(lut_names) - 1)])


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("🎛️ LUT Library")

    presets = load_presets()

    def preset_label(p: dict) -> str:
        scene = p.get("scene", "?")
        tgt = p.get("saved_target", "LUT A")
        primary = p.get("lut_primary") or p.get("lut_a_name") or ""
        return f"Scene {scene} — {tgt} — {primary}"

    preset_options = ["(none)"] + [preset_label(p) for p in presets]
    st.session_state.setdefault("preset_idx", 0)

    def on_preset_change():
        idx = st.session_state["preset_idx"]
        if idx and 1 <= idx <= len(presets):
            queue_apply_preset(presets[idx - 1])

    st.selectbox(
        "Presets",
        options=list(range(len(preset_options))),
        format_func=lambda i: preset_options[i],
        key="preset_idx",
        on_change=on_preset_change,
    )

    st.divider()

    st.subheader("➕ Add LUTs (.cube)")
    uploaded_luts = st.file_uploader(
        "Upload one or more .cube files (available this session; also tries saving to /luts)",
        type=["cube"],
        accept_multiple_files=True,
    )

    if uploaded_luts:
        LUT_DIR.mkdir(parents=True, exist_ok=True)
        for f in uploaded_luts:
            b = f.getvalue()
            st.session_state["uploaded_luts"][f.name] = b
            try:
                with open(LUT_DIR / f.name, "wb") as w:
                    w.write(b)
            except Exception:
                pass

        st.success(f"Loaded {len(uploaded_luts)} LUT(s).")
        st.cache_data.clear()
        st.rerun()

    st.divider()

    st.checkbox("If colors look wrong, try alternate .cube order", key="assume_order")
    assume_bgr_major = not st.session_state["assume_order"]

    lut_map = get_lut_map()
    lut_names = sorted(lut_map.keys())

    st.caption(f"LUT folder: `{LUT_DIR}`")
    st.caption(f"Found **{len(lut_names)}** LUT(s).")
    st.caption(f"Example image: `{EXAMPLE_IMG_PATH}`" if EXAMPLE_IMG_PATH else "Example image: (not found)")

    st.divider()
    st.subheader("🖼️ Previews (one-click select)")

    if EXAMPLE_IMG_PATH and EXAMPLE_IMG_PATH.exists() and lut_map:
        ex_pil = Image.open(EXAMPLE_IMG_PATH).convert("RGB")
        ex_rgb = np.array(ex_pil, dtype=np.uint8)

        max_w = 420
        if ex_pil.width > max_w:
            scale = max_w / ex_pil.width
            ex_pil = ex_pil.resize((int(ex_pil.width * scale), int(ex_pil.height * scale)))
            ex_rgb = np.array(ex_pil, dtype=np.uint8)

        for name, b in sorted(lut_map.items(), key=lambda kv: kv[0].lower()):
            with st.expander(name, expanded=False):
                try:
                    lut = load_cube_lut(b, assume_bgr_major=assume_bgr_major)
                    applied = apply_lut_trilinear(ex_rgb, lut, strength=1.0)
                    st.image(applied, caption="Applied on example image", use_container_width=True)

                    r = st.columns([1, 1])
                    with r[0]:
                        if st.button("✅ Set as LUT A", key=f"setA_{name}"):
                            st.session_state["lut_a_name"] = name
                            st.rerun()
                    with r[1]:
                        if st.button("⚖️ Compare (set as LUT B)", key=f"setB_{name}"):
                            st.session_state["compare_mode"] = "LUT A vs LUT B"
                            st.session_state["lut_b_name"] = name
                            st.rerun()
                except Exception as e:
                    st.warning(f"Could not preview {name}: {e}")
    else:
        st.info("To enable sidebar thumbnails: add `assets/example.png` (either repo root/assets or next to app.py/assets).")


# -----------------------------
# Controls row (Row 1)
# -----------------------------
controls = st.container()
with controls:
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])

    with c1:
        video_file = st.file_uploader("Upload video", type=["mp4", "mov", "mkv", "avi", "webm"])

    with c2:
        lut_map = get_lut_map()
        lut_names = sorted(lut_map.keys())

        if not lut_names:
            st.error("No LUTs available. Add .cube files into /luts (repo root or next to app.py), or upload them in the sidebar.")
            st.stop()

        if st.session_state.get("lut_a_name") not in lut_names:
            st.session_state["lut_a_name"] = lut_names[0]
        if st.session_state.get("lut_b_name") not in lut_names:
            st.session_state["lut_b_name"] = lut_names[min(1, len(lut_names) - 1)]

        st.selectbox("LUT A", lut_names, key="lut_a_name")
        st.selectbox("Compare mode", ["LUT A vs Original", "LUT A vs LUT B"], key="compare_mode")
        if st.session_state["compare_mode"] == "LUT A vs LUT B":
            st.selectbox("LUT B", lut_names, key="lut_b_name")

    with c3:
        st.slider("Strength A", 0.0, 1.0, 1.0, 0.01, key="strength_a")
        if st.session_state["compare_mode"] == "LUT A vs LUT B":
            st.slider("Strength B", 0.0, 1.0, 1.0, 0.01, key="strength_b")

    with c4:
        with st.popover("💾 Save Scene Preset"):
            scene = st.text_input("Scene #", value="")
            preset_name = st.text_input("Preset name (optional)", value="")

            if st.session_state["compare_mode"] == "LUT A vs LUT B":
                target = st.radio("Save which one?", ["LUT A", "LUT B"], horizontal=True)
            else:
                target = "LUT A"
                st.caption("Compare mode is off, saving LUT A preset.")

            if st.button("Save preset"):
                if not scene.strip():
                    st.error("Please enter a Scene #.")
                else:
                    lut_a = st.session_state["lut_a_name"]
                    lut_b = st.session_state.get("lut_b_name")
                    strength_a = float(st.session_state["strength_a"])
                    strength_b = float(st.session_state.get("strength_b", 1.0))

                    entry = {
                        "scene": scene.strip(),
                        "name": preset_name.strip() or f"Scene {scene.strip()}",
                        "saved_target": target,
                        "compare_mode": st.session_state["compare_mode"],
                        "lut_a_name": lut_a,
                        "strength_a": strength_a,
                        "assume_order": bool(st.session_state["assume_order"]),
                    }

                    if st.session_state["compare_mode"] == "LUT A vs LUT B":
                        entry["lut_b_name"] = lut_b
                        entry["strength_b"] = strength_b
                        if target == "LUT B":
                            entry["lut_primary"] = lut_b
                            entry["strength_primary"] = strength_b
                        else:
                            entry["lut_primary"] = lut_a
                            entry["strength_primary"] = strength_a
                    else:
                        entry["lut_primary"] = lut_a
                        entry["strength_primary"] = strength_a

                    save_preset(entry)
                    st.success("Preset saved.")
                    st.rerun()

if not video_file:
    st.info("Upload a video to begin.")
    st.stop()

assume_bgr_major = not st.session_state["assume_order"]

# -----------------------------
# Load video + frames
# -----------------------------
video_path = _write_uploaded_to_temp(video_file)
frames_bgr, meta = extract_three_frames(video_path)
frames_rgb = [bgr_to_rgb_u8(f) for f in frames_bgr]

st.caption(f"Frames: start={meta['idxs'][0]}, mid={meta['idxs'][1]}, end={meta['idxs'][2]} | fps≈{meta['fps']:.2f}")

lut_map = get_lut_map()


def apply_named(rgb: np.ndarray, lut_name: str, strength: float) -> np.ndarray:
    lut = load_cube_lut(lut_map[lut_name], assume_bgr_major=assume_bgr_major)
    return apply_lut_trilinear(rgb, lut, strength=strength)


# -----------------------------
# Previews (Row 2) -> 3 columns
# -----------------------------
st.subheader("Preview (Start / Middle / End)")
p1, p2, p3 = st.columns(3)

lut_a = st.session_state["lut_a_name"]
strength_a = float(st.session_state["strength_a"])
compare_mode = st.session_state["compare_mode"]
lut_b = st.session_state.get("lut_b_name")
strength_b = float(st.session_state.get("strength_b", 1.0))

for i, (col, label, rgb) in enumerate(zip([p1, p2, p3], ["Start", "Middle", "End"], frames_rgb)):
    with col:
        st.markdown(f"**{label}**")
        if compare_mode == "LUT A vs Original":
            out = apply_named(rgb, lut_a, strength_a)
            img1 = save_rgb_to_png_path(rgb, f"orig_{label}_{i}")
            img2 = save_rgb_to_png_path(out, f"lutA_{label}_{i}")
            image_comparison(img1=img1, img2=img2, label1="Original", label2=f"{lut_a} ({strength_a:.2f})")
        else:
            a_img = apply_named(rgb, lut_a, strength_a)
            b_img = apply_named(rgb, lut_b, strength_b)
            img1 = save_rgb_to_png_path(a_img, f"lutA_{label}_{i}")
            img2 = save_rgb_to_png_path(b_img, f"lutB_{label}_{i}")
            image_comparison(
                img1=img1,
                img2=img2,
                label1=f"{lut_a} ({strength_a:.2f})",
                label2=f"{lut_b} ({strength_b:.2f})",
            )

st.divider()

# -----------------------------
# Export
# -----------------------------
st.subheader("Export")
safe_lut = (lut_a or "none").replace(" ", "_")
out_name = st.text_input("Output filename", value=f"export_{Path(video_file.name).stem}_{safe_lut}.mp4")

from shutil import which


def get_ffmpeg_path() -> Optional[str]:
    env = os.environ.get("FFMPEG_PATH")
    if env and Path(env).exists():
        return str(Path(env))

    p = which("ffmpeg")
    if p:
        return p

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def export_video_with_lut_opencv(input_path: str, output_path: str, lut_name: str, strength: float, assume_bgr_major: bool):
    """Slow fallback export using OpenCV + our LUT implementation."""
    lut = load_cube_lut(lut_map[lut_name], assume_bgr_major=assume_bgr_major)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Could not open video for export.")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_out = apply_lut_trilinear(rgb, lut, strength=strength)
        bgr_out = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
        writer.write(bgr_out)

    cap.release()
    writer.release()


def export_video_with_lut(input_path: str, output_path: str, lut_name: str, strength: float) -> str:
    """
    Preferred export path:
      - If ffmpeg is available: apply LUT via ffmpeg lut3d filter (fast) and keep audio.
      - If strength < 1.0: we blend via ffmpeg (format+blend) for a true "strength" effect.
      - If ffmpeg missing: fallback to OpenCV (video-only), audio merged later if possible.
    Returns path to a video file containing processed video (may or may not include audio).
    """
    ffmpeg = get_ffmpeg_path()
    lut_file = LUT_DIR / lut_name

    # If LUT is only in session (uploaded), write it to disk so ffmpeg can read it
    if not lut_file.exists() and lut_name in st.session_state.get("uploaded_luts", {}):
        try:
            LUT_DIR.mkdir(parents=True, exist_ok=True)
            lut_file.write_bytes(st.session_state["uploaded_luts"][lut_name])
        except Exception:
            pass

        if ffmpeg and lut_file.exists():
            lut_path = str(lut_file).replace("\\", "/")   # ffmpeg friendly on Windows too
            lut_path_esc = lut_path.replace("'", r"\'")   # escape single quotes for ffmpeg

            # -------------------------
            # Strength == 1.0 (fast path)
            # -------------------------
            if strength >= 0.999:
                cmd = [
                    ffmpeg, "-y",
                    "-i", input_path,
                    "-vf", f"lut3d=file='{lut_path_esc}'",
                    "-c:v", "libx264",
                    "-crf", "18",
                    "-preset", "veryfast",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "copy",
                    "-movflags", "+faststart",
                    output_path,
                ]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if Path(output_path).exists() and Path(output_path).stat().st_size > 0 and r.returncode == 0:
                    return output_path

            # -------------------------
            # Strength < 1.0 (blend path)
            # -------------------------
            alpha = float(np.clip(strength, 0.0, 1.0))
            filter_complex = (
                f"[0:v]format=rgba[orig];"
                f"[0:v]lut3d=file='{lut_path_esc}',format=rgba[lut];"
                f"[orig][lut]blend=all_mode=normal:all_opacity={alpha},format=yuv420p[outv]"
            )

            cmd = [
                ffmpeg, "-y",
                "-i", input_path,
                "-filter_complex", filter_complex,
                "-map", "[outv]",
                "-map", "0:a:0?",
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "veryfast",
                "-c:a", "aac",
                "-b:a", "192k",
                "-ac", "2",
                "-movflags", "+faststart",
                output_path,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if Path(output_path).exists() and Path(output_path).stat().st_size > 0 and r.returncode == 0:
                return output_path

            with st.expander("ffmpeg export failed — logs"):
                st.code(r.stderr or r.stdout or "no output")


    # Fallback (OpenCV video-only)
    export_video_with_lut_opencv(
        input_path=input_path,
        output_path=output_path,
        lut_name=lut_name,
        strength=strength,
        assume_bgr_major=assume_bgr_major,
    )
    return output_path


def merge_audio(processed_video: str, original_video: str, out_path: str) -> str:
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        st.warning("ffmpeg not found, exporting without audio.")
        return processed_video

    # 1) Attempt: copy audio streams (fast)
    cmd_copy = [
        ffmpeg, "-y",
        "-i", processed_video,
        "-i", original_video,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "copy",
        "-shortest",
        "-movflags", "+faststart",
        out_path,
    ]
    r1 = subprocess.run(cmd_copy, capture_output=True, text=True)

    if Path(out_path).exists() and Path(out_path).stat().st_size > 0 and r1.returncode == 0:
        return out_path

    # 2) Fallback: re-encode audio to AAC
    out_path2 = str(Path(out_path).with_name("video_with_audio_aac.mp4"))
    cmd_aac = [
        ffmpeg, "-y",
        "-i", processed_video,
        "-i", original_video,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ac", "2",
        "-shortest",
        "-movflags", "+faststart",
        out_path2,
    ]
    r2 = subprocess.run(cmd_aac, capture_output=True, text=True)

    if Path(out_path2).exists() and Path(out_path2).stat().st_size > 0 and r2.returncode == 0:
        return out_path2

    with st.expander("Audio merge failed — ffmpeg logs"):
        st.code("COPY attempt:\n" + (r1.stderr or r1.stdout or "no output"))
        st.code("AAC attempt:\n" + (r2.stderr or r2.stdout or "no output"))

    st.warning("ffmpeg merge failed, exporting without audio.")
    return processed_video


def local_save_dialog(default_name: str, src_path: str) -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)

        save_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            initialfile=default_name,
            filetypes=[("MP4 Video", "*.mp4"), ("All files", "*.*")],
        )
        root.destroy()

        if save_path:
            shutil.copyfile(src_path, save_path)
            return save_path
        return None
    except Exception:
        return None


if st.button("🚀 Export LUT A applied video"):
    with st.spinner("Exporting..."):
        tmp_dir = tempfile.mkdtemp()
        raw_out = os.path.join(tmp_dir, "video_processed.mp4")

        # ✅ this is now defined (and prefers ffmpeg path)
        processed_path = export_video_with_lut(video_path, raw_out, lut_a, strength_a)

        # If processed export already includes audio, merge_audio will just re-copy safely
        final_out = merge_audio(processed_path, video_path, os.path.join(tmp_dir, "video_with_audio.mp4"))
        st.session_state["last_export_path"] = final_out
        st.session_state["last_export_name"] = out_name

    st.success("Export complete.")


if "last_export_path" in st.session_state and Path(st.session_state["last_export_path"]).exists():
    export_path = st.session_state["last_export_path"]
    export_name = st.session_state.get("last_export_name", "export.mp4")

    with open(export_path, "rb") as f:
        st.download_button("⬇️ Download exported video", data=f.read(), file_name=export_name, mime="video/mp4")

    if ENABLE_LOCAL_SAVE_DIALOG and (not running_on_streamlit_cloud()) and gui_available():
        if st.button("💾 Save As… (local)"):
            saved = local_save_dialog(export_name, export_path)
            if saved:
                st.success(f"Saved to: {saved}")
            else:
                st.info("Save cancelled (or dialog unavailable).")
    else:
        st.caption("Note: browsers control download location; use the download prompt, or run locally for Save As…")
        
