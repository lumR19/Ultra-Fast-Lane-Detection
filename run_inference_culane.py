import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import sys
import time
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '.')
from model.model import parsingNet

# ══════════════════════════════════════════════════════════════════════════════
#   UFLD Inference — CULane pretrained model
#   General purpose version — works with any video automatically
# ══════════════════════════════════════════════════════════════════════════════

# ── ONLY CHANGE THESE 2 LINES FOR EACH NEW VIDEO ─────────────────────────────
WEIGHTS   = r'C:\Users\Lenovo\Desktop\Graduation_tammakan\ufld_project\Ultra-Fast-Lane-Detection\culane_18.pth'
VIDEO_IN  = r'C:\Users\Lenovo\Desktop\Graduation_tammakan\ufld_project\Ultra-Fast-Lane-Detection\dataset_input_forTEST\2video_test_night.mp4'
# ─────────────────────────────────────────────────────────────────────────────

# Output video is auto-named based on input filename — no need to change it
# Save output to your specific folder
OUTPUT_DIR = r'C:\Users\Lenovo\Desktop\Graduation_tammakan\ufld_project\Ultra-Fast-Lane-Detection\test_output_Culane'
os.makedirs(OUTPUT_DIR, exist_ok=True)  # creates the folder if it doesn't exist

# Get just the filename from input video, save it in the output folder
VIDEO_OUT = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(VIDEO_IN))[0] + '_NEW_V_culane_output.mp4')

# ── MODEL SETTINGS (fixed — do not change) ────────────────────────────────────
BACKBONE     = '18'
IMG_W, IMG_H = 800, 288
GRID_W       = 200

CULANE_ROW_ANCHORS = [
    80,  93, 106, 119, 132, 145, 158, 171, 184,
    197, 210, 223, 236, 249, 262, 275, 275, 280
]

# Column sample scaled from training size (1640px) to model input (800px)
col_sample = np.linspace(0, 1640 - 1, GRID_W) * (IMG_W / 1640)

COLORS = [
    ( 80,  80, 255),   # Lane 0 — blue
    ( 80, 255,  80),   # Lane 1 — green
    (255, 150,  80),   # Lane 2 — orange
    ( 80, 220, 255),   # Lane 3 — cyan
]

# ── ADAPTIVE SETTINGS — auto-calculated per video ─────────────────────────────
# These are calculated automatically from the video brightness
# so the same script works for day, night, tunnel, highway, urban

# Confidence: starts at 0.15, auto-lowers to 0.08 if scene is dark
BASE_CONFIDENCE  = 0.15
DARK_CONFIDENCE  = 0.08   # used when avg brightness < 80
DARK_THRESHOLD   = 80     # brightness below this = night/tunnel scene

# ROI: top 35% of image = sky/ceiling, bottom 10% = car hood
# These are fractions of IMG_H so they work for any resolution
ROI_TOP_FRAC    = 0.35   # ignore top 35% of image
ROI_BOTTOM_FRAC = 0.92   # ignore bottom 8% of image

# Minimum points per lane — higher = stricter, fewer false lanes
MIN_POINTS = 4

# Curve clipping — how far the smooth curve can go beyond detected points
CURVE_MARGIN = 40   # pixels

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
print("="*60)
print("  UFLD — CULane  |  General Purpose Inference")
print("="*60)
print("\nLoading model...")

net = parsingNet(
    pretrained=False,
    backbone=BACKBONE,
    cls_dim=(GRID_W + 1, len(CULANE_ROW_ANCHORS), 4),
    use_aux=False
)

state_dict = torch.load(WEIGHTS, map_location='cpu')
if 'model' in state_dict:
    state_dict = state_dict['model']

compatible_state_dict = {}
for k, v in state_dict.items():
    compatible_state_dict[k[7:] if 'module.' in k else k] = v

net.load_state_dict(compatible_state_dict, strict=False)
net.eval()
print("Model loaded!\n")

# ── IMAGE TRANSFORM ───────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

# ── OPEN VIDEO ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_IN}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps          = cap.get(cv2.CAP_PROP_FPS)
orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate ROI pixel values from fractions
ROI_TOP    = int(IMG_H * ROI_TOP_FRAC)     # e.g. 288 * 0.35 = ~100px
ROI_BOTTOM = int(IMG_H * ROI_BOTTOM_FRAC)  # e.g. 288 * 0.92 = ~265px

print(f"Video        : {os.path.basename(VIDEO_IN)}")
print(f"Frames       : {total_frames} | {fps} FPS | {orig_w}x{orig_h}")
print(f"Model input  : {IMG_W}x{IMG_H}")
print(f"ROI          : y={ROI_TOP} to y={ROI_BOTTOM} (auto-calculated)")
print(f"Output       : {os.path.basename(VIDEO_OUT)}\n")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (IMG_W, IMG_H))

# ── METRICS STORAGE ───────────────────────────────────────────────────────────
frame_times       = []
lanes_per_frame   = []
confidence_scores = []
points_per_lane   = []
brightness_values = []   # track scene brightness over time


# ══════════════════════════════════════════════════════════════════════════════
#   HELPER 1: detect scene brightness and return adaptive confidence threshold
# ══════════════════════════════════════════════════════════════════════════════
def get_adaptive_confidence(frame):
    """
    Measures the average brightness of the road region.
    Dark scenes (night/tunnel) need a lower confidence threshold
    because lane markings produce weaker model responses.
    Bright scenes (daytime) can use a higher threshold to avoid
    false detections from road texture and shadows.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Only measure brightness in the ROI road area — not sky or hood
    road_region = gray[ROI_TOP:ROI_BOTTOM, :]
    brightness  = float(np.mean(road_region))
    brightness_values.append(brightness)

    if brightness < DARK_THRESHOLD:
        return DARK_CONFIDENCE   # night or tunnel — be more sensitive
    else:
        return BASE_CONFIDENCE   # daytime — be more strict


# ══════════════════════════════════════════════════════════════════════════════
#   HELPER 2: draw smooth polynomial curve through detected points
# ══════════════════════════════════════════════════════════════════════════════
def draw_smooth_lane(canvas, xs, ys, color):
    """
    Fits a 2nd degree polynomial through detected lane points
    and draws a smooth curve. Includes curve clipping to prevent
    wild arcs when points are noisy.
    """
    if len(xs) < 3:
        for x, y in zip(xs, ys):
            cv2.circle(canvas, (int(x), int(y)), 4, color, -1)
        return

    xs_arr = np.array(xs, dtype=np.float32)
    ys_arr = np.array(ys, dtype=np.float32)

    try:
        # Fit x = f(y) polynomial — deg=2 for smooth curves
        coeffs = np.polyfit(ys_arr, xs_arr, deg=2)

        # Generate smooth points every 2 pixels
        y_smooth = np.arange(
            max(ROI_TOP,    int(np.min(ys_arr))),
            min(ROI_BOTTOM, int(np.max(ys_arr)) + 1),
            2
        ).astype(np.float32)

        if len(y_smooth) < 2:
            return

        x_smooth = np.polyval(coeffs, y_smooth)

        # ── Curve clipping: stop curve from going beyond detected range ───────
        # This prevents wild parabolic arcs from noisy points
        x_min = np.min(xs_arr) - CURVE_MARGIN
        x_max = np.max(xs_arr) + CURVE_MARGIN
        valid    = (x_smooth >= x_min) & (x_smooth <= x_max)
        x_smooth = x_smooth[valid]
        y_smooth = y_smooth[valid]

        if len(y_smooth) < 2:
            return
        # ─────────────────────────────────────────────────────────────────────

        # Clip to image boundaries
        x_smooth = np.clip(x_smooth, 0, IMG_W - 1).astype(np.int32)
        y_smooth = y_smooth.astype(np.int32)

        # Draw smooth curve
        pts = np.stack([x_smooth, y_smooth], axis=1).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=4)

        # Draw clean dots at actual detected anchor positions
        for x, y in zip(xs, ys):
            cv2.circle(canvas, (int(x), int(y)), 3, color, -1)

    except Exception:
        # Fallback: raw polyline if polynomial fails
        pts = np.array(list(zip(xs, ys)), dtype=np.int32)
        cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, color, 3)


# ══════════════════════════════════════════════════════════════════════════════
#   MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
video_start = time.time()
frame_idx   = 0
print("Processing...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video finished!")
        break

    frame_idx += 1
    frame_resized = cv2.resize(frame, (IMG_W, IMG_H))

    # ── Auto-detect scene type and set confidence threshold ───────────────────
    confidence = get_adaptive_confidence(frame_resized)

    # ── Normalize and run inference ───────────────────────────────────────────
    img_rgb      = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    input_tensor = transform(Image.fromarray(img_rgb)).unsqueeze(0)

    t0 = time.time()
    with torch.no_grad():
        output = net(input_tensor)
    inference_ms = (time.time() - t0) * 1000
    frame_times.append(inference_ms)

    # ── Decode output ─────────────────────────────────────────────────────────
    output_np   = output[0].numpy()
    scores_np   = output_np[:GRID_W, :, :]
    lane_points = np.argmax(scores_np, axis=0)
    max_scores  = np.max(scores_np, axis=0)
    exist_flags = max_scores > confidence   # ← uses adaptive threshold

    # ── Draw lanes ────────────────────────────────────────────────────────────
    canvas           = frame_resized.copy()
    lanes_this_frame = 0
    elapsed          = time.time() - video_start
    avg_fps          = frame_idx / elapsed if elapsed > 0 else 0
    eta              = (total_frames - frame_idx) / avg_fps if avg_fps > 0 else 0

    # Subtle ROI lines so you can see the detection zone
    cv2.line(canvas, (0, ROI_TOP),    (IMG_W, ROI_TOP),    (50, 50, 50), 1)
    cv2.line(canvas, (0, ROI_BOTTOM), (IMG_W, ROI_BOTTOM), (50, 50, 50), 1)

    for lane_idx in range(4):
        xs, ys, confs = [], [], []

        for row_idx, row_anchor in enumerate(CULANE_ROW_ANCHORS):
            # Skip points outside ROI
            if not (ROI_TOP <= int(row_anchor) <= ROI_BOTTOM):
                continue

            if exist_flags[row_idx, lane_idx]:
                x = int(col_sample[lane_points[row_idx, lane_idx]])
                y = int(row_anchor)
                xs.append(x)
                ys.append(y)
                confs.append(float(max_scores[row_idx, lane_idx]))

        # Only accept lanes with enough confident points
        if len(xs) > MIN_POINTS:
            lanes_this_frame += 1
            confidence_scores.append(np.mean(confs))
            points_per_lane.append(len(xs))
            draw_smooth_lane(canvas, xs, ys, COLORS[lane_idx])

    lanes_per_frame.append(lanes_this_frame)

    # ── Scene type label (auto-detected) ──────────────────────────────────────
    brightness   = brightness_values[-1]
    scene_label  = "Night/Tunnel" if brightness < DARK_THRESHOLD else "Daytime"

    # ── Overlay stats ─────────────────────────────────────────────────────────
    overlay = [
        f"Frame: {frame_idx}/{total_frames}",
        f"Inference: {inference_ms:.0f} ms  ({1000/inference_ms:.1f} FPS)",
        f"Avg FPS: {avg_fps:.2f}",
        f"ETA: {int(eta//60)}m {int(eta%60)}s",
        f"Lanes: {lanes_this_frame} | Scene: {scene_label}",
        f"Brightness: {brightness:.0f} | Conf: {confidence:.2f}",
    ]
    for i, txt in enumerate(overlay):
        cv2.putText(canvas, txt, (10, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3)
        cv2.putText(canvas, txt, (10, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

    out.write(canvas)

    if frame_idx % 25 == 0:
        print(f"  Frame {frame_idx:>4}/{total_frames} | "
              f"{inference_ms:>5.0f} ms | "
              f"Lanes: {lanes_this_frame} | "
              f"Scene: {scene_label:<13} | "
              f"ETA: {int(eta//60)}m {int(eta%60)}s")

# ── CLEANUP ───────────────────────────────────────────────────────────────────
cap.release()
out.release()
total_time = time.time() - video_start

# ── FINAL REPORT ──────────────────────────────────────────────────────────────
no_lane_frames    = lanes_per_frame.count(0)
one_lane_frames   = sum(1 for x in lanes_per_frame if x == 1)
two_lane_frames   = sum(1 for x in lanes_per_frame if x == 2)
three_lane_frames = sum(1 for x in lanes_per_frame if x >= 3)
night_frames      = sum(1 for b in brightness_values if b < DARK_THRESHOLD)
day_frames        = total_frames - night_frames

print("\n" + "="*60)
print("        EVALUATION REPORT — UFLD CULane General")
print("="*60)
print(f"\n  [Video Info]")
print(f"  File                     : {os.path.basename(VIDEO_IN)}")
print(f"  Duration                 : {total_frames/fps:.1f} sec")
print(f"  Daytime frames           : {day_frames} ({100*day_frames/total_frames:.1f}%)")
print(f"  Night/Tunnel frames      : {night_frames} ({100*night_frames/total_frames:.1f}%)")
print(f"  Avg scene brightness     : {np.mean(brightness_values):.1f}/255")

print(f"\n  [Speed Metrics]")
print(f"  Total processing time    : {total_time:.1f} sec ({total_time/60:.1f} min)")
print(f"  Avg inference per frame  : {np.mean(frame_times):.1f} ms")
print(f"  Min inference time       : {np.min(frame_times):.1f} ms")
print(f"  Max inference time       : {np.max(frame_times):.1f} ms")
print(f"  Avg processing speed     : {total_frames/total_time:.2f} FPS")
print(f"  Real-time capable (CPU)  : {'YES' if total_frames/total_time >= fps else 'NO (expected on CPU)'}")

print(f"\n  [Detection Metrics]")
print(f"  Total frames processed   : {total_frames}")
print(f"  Avg lanes per frame      : {np.mean(lanes_per_frame):.2f}")
if confidence_scores:
    print(f"  Avg confidence score     : {min(np.mean(confidence_scores)/100, 1.0):.4f}")
    print(f"  Avg points per lane      : {np.mean(points_per_lane):.1f}")
print(f"  Frames with 0 lanes      : {no_lane_frames} ({100*no_lane_frames/total_frames:.1f}%)")
print(f"  Frames with 1 lane       : {one_lane_frames} ({100*one_lane_frames/total_frames:.1f}%)")
print(f"  Frames with 2 lanes      : {two_lane_frames} ({100*two_lane_frames/total_frames:.1f}%)")
print(f"  Frames with 3+ lanes     : {three_lane_frames} ({100*three_lane_frames/total_frames:.1f}%)")

print(f"\n  [Published Benchmark — CULane dataset, from paper]")
print(f"  F1 Score (overall)       : 68.4%")
print(f"  F1 Score (night)         : 66.3%")
print(f"  F1 Score (crowded)       : 69.7%")
print(f"  F1 Score (no line)       : 41.7%")
print(f"  Speed (GPU)              : 322 FPS")
print("="*60)
print(f"\n  Output saved to: {VIDEO_OUT}")