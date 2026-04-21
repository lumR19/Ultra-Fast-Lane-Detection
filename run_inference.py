import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import sys
import time

sys.path.insert(0, '.')
from model.model import parsingNet
# this code is for testing the model that was pretrained on tusimple dataset.
# ── SETTINGS ──────────────────────────────────────────────────────────────────
# Don't forget to change the pathes here IMPORTAT only for the input & output videos
BACKBONE  = '18'
WEIGHTS   = r'C:\Users\Lenovo\Desktop\Graduation_tammakan\ufld_project\Ultra-Fast-Lane-Detection\tusimple_18.pth'
VIDEO_IN  = r'C:\Users\Lenovo\Desktop\Graduation_tammakan\ufld_project\Ultra-Fast-Lane-Detection\3video_test_daylight.mp4'
VIDEO_OUT = r'C:\Users\Lenovo\Desktop\Graduation_tammakan\ufld_project\Ultra-Fast-Lane-Detection\3output_lanes.mp4'

TUSIMPLE_ROW_ANCHORS = [
    64,  68,  72,  76,  80,  84,  88,  92,  96, 100,
   104, 108, 112, 116, 120, 124, 128, 132, 136, 140,
   144, 148, 152, 156, 160, 164, 168, 172, 176, 180,
   184, 188, 192, 196, 200, 204, 208, 212, 216, 220,
   224, 228, 232, 236, 240, 244, 248, 252, 256, 260,
   264, 268, 272, 276, 280, 284
]
IMG_W, IMG_H = 800, 288
GRID_W       = 100
CONFIDENCE   = 0.2

COLORS = [
    (255,  80,  80),
    ( 80, 255,  80),
    ( 80, 150, 255),
    (255, 200,  50),
]

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
print("Loading model...")
net = parsingNet(pretrained=False, backbone=BACKBONE,
                 cls_dim=(101, 56, 4), use_aux=False)
state_dict = torch.load(WEIGHTS, map_location='cpu')
if 'model' in state_dict:
    state_dict = state_dict['model']
net.load_state_dict(state_dict, strict=False)
net.eval()
print("Model loaded!\n")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

# ── OPEN VIDEO ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_IN)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps          = cap.get(cv2.CAP_PROP_FPS)
orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {total_frames} frames | {fps} FPS | {orig_w}x{orig_h}\n")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (IMG_W, IMG_H))
col_sample = np.linspace(0, IMG_W - 1, GRID_W)

# ── METRICS STORAGE ───────────────────────────────────────────────────────────
frame_times           = []
lanes_per_frame       = []
confidence_scores     = []   # average confidence per detected lane point
points_per_lane       = []   # how many points each detected lane has

# ── PROCESS ───────────────────────────────────────────────────────────────────
video_start = time.time()
frame_idx   = 0
print("Processing... (be patient on CPU)\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    frame_resized = cv2.resize(frame, (IMG_W, IMG_H))
    img_rgb       = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    input_tensor  = transform(Image.fromarray(img_rgb)).unsqueeze(0)

    # ── Inference ─────────────────────────────────────────────────────────────
    t0 = time.time()
    with torch.no_grad():
        output = net(input_tensor)
    inference_ms = (time.time() - t0) * 1000
    frame_times.append(inference_ms)

    # ── Decode ────────────────────────────────────────────────────────────────
    output_np    = output[0].numpy()
    scores_np    = output_np[:GRID_W, :, :]   # (100, 56, 4)
    lane_points  = np.argmax(scores_np, axis=0)
    max_scores   = np.max(scores_np, axis=0)   # confidence per point
    exist_flags  = max_scores > CONFIDENCE

    canvas           = frame_resized.copy()
    lanes_this_frame = 0
    elapsed          = time.time() - video_start
    avg_fps          = frame_idx / elapsed if elapsed > 0 else 0
    eta              = (total_frames - frame_idx) / avg_fps if avg_fps > 0 else 0

    for lane_idx in range(4):
        xs, ys, confs = [], [], []
        for row_idx, row_anchor in enumerate(TUSIMPLE_ROW_ANCHORS):
            if exist_flags[row_idx, lane_idx]:
                x = int(col_sample[lane_points[row_idx, lane_idx]])
                y = int(row_anchor)
                xs.append(x)
                ys.append(y)
                confs.append(float(max_scores[row_idx, lane_idx]))

        if len(xs) > 5:
            lanes_this_frame += 1
            confidence_scores.append(np.mean(confs))
            points_per_lane.append(len(xs))

            for x, y in zip(xs, ys):
                cv2.circle(canvas, (x, y), 4, COLORS[lane_idx], -1)
            pts = np.array(list(zip(xs, ys)), dtype=np.int32)
            cv2.polylines(canvas, [pts.reshape(-1, 1, 2)],
                          False, COLORS[lane_idx], 3)

    lanes_per_frame.append(lanes_this_frame)

    # ── Overlay on frame ──────────────────────────────────────────────────────
    lines = [
        f"Frame: {frame_idx}/{total_frames}",
        f"Inference: {inference_ms:.0f} ms  ({1000/inference_ms:.1f} FPS)",
        f"Avg FPS: {avg_fps:.2f}",
        f"ETA: {int(eta//60)}m {int(eta%60)}s",
        f"Lanes detected: {lanes_this_frame}",
    ]
    for i, txt in enumerate(lines):
        cv2.putText(canvas, txt, (10, 25 + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3)
        cv2.putText(canvas, txt, (10, 25 + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    out.write(canvas)

    if frame_idx % 25 == 0:
        print(f"  Frame {frame_idx:>4}/{total_frames} | "
              f"{inference_ms:>5.0f} ms | "
              f"Lanes: {lanes_this_frame} | "
              f"ETA: {int(eta//60)}m {int(eta%60)}s")

cap.release()
out.release()
total_time = time.time() - video_start

# ── FINAL REPORT ──────────────────────────────────────────────────────────────
no_lane_frames   = lanes_per_frame.count(0)
one_lane_frames  = sum(1 for x in lanes_per_frame if x == 1)
two_lane_frames  = sum(1 for x in lanes_per_frame if x == 2)
three_lane_frames= sum(1 for x in lanes_per_frame if x >= 3)

print("\n" + "="*60)
print("            EVALUATION REPORT — UFLD on Custom Video")
print("="*60)
print(f"\n  [Speed Metrics]")
print(f"  Video duration           : {total_frames/fps:.1f} sec")
print(f"  Total processing time    : {total_time:.1f} sec ({total_time/60:.1f} min)")
print(f"  Avg inference per frame  : {np.mean(frame_times):.1f} ms")
print(f"  Min inference time       : {np.min(frame_times):.1f} ms")
print(f"  Max inference time       : {np.max(frame_times):.1f} ms")
print(f"  Avg processing speed     : {total_frames/total_time:.2f} FPS")
print(f"  Real-time capable (CPU)  : {'YES' if total_frames/total_time >= fps else 'NO (expected on CPU)'}")

print(f"\n  [Detection Metrics]")
print(f"  Total frames processed   : {total_frames}")
print(f"  Avg lanes per frame      : {np.mean(lanes_per_frame):.2f}")
print(f"  Avg confidence score     : {np.mean(confidence_scores):.4f}" if confidence_scores else "  Avg confidence score     : N/A")
print(f"  Avg points per lane      : {np.mean(points_per_lane):.1f}" if points_per_lane else "  Avg points per lane      : N/A")
print(f"  Frames with 0 lanes      : {no_lane_frames} ({100*no_lane_frames/total_frames:.1f}%)")
print(f"  Frames with 1 lane       : {one_lane_frames} ({100*one_lane_frames/total_frames:.1f}%)")
print(f"  Frames with 2 lanes      : {two_lane_frames} ({100*two_lane_frames/total_frames:.1f}%)")
print(f"  Frames with 3+ lanes     : {three_lane_frames} ({100*three_lane_frames/total_frames:.1f}%)")

print(f"\n  [Published Benchmark — TuSimple dataset, from paper]")
print(f"  Accuracy (ACC)           : 95.87%")
print(f"  False Positive (FP)      : 2.62%")
print(f"  False Negative (FN)      : 2.33%")
print(f"  F1 Score                 : 96.52%")
print(f"  Speed (GPU)              : 322 FPS")
print("="*60)
print(f"\n  Output saved to: {VIDEO_OUT}")