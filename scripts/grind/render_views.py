#!/usr/bin/env python3
"""Render a baked press recording to video from several fixed points of view.

Replay in the GUI is one camera at a time and nobody can review five parts that way. This walks
each recording headless, once per named view, captures the viewport frame by frame and encodes an
H.264 mp4. No physics runs — it is the same baked animation the interactive replay shows.

    python scripts/grind/render_views.py runs/part_* --out runs/part_videos

Views are computed from each part's own bounds, so the same name frames every part comparably:

    iso      three-quarter view, the default overview
    across   low and square to the band — shows the disc riding a shoulder beside the line
    along    low and down the line, looking along travel — shows the lead tilt
    top      plan view — shows the patch against the green marked region
    chase    tracks the disc down the pass, close in

Needs imageio-ffmpeg (`uv pip install --python .venv/bin/python imageio-ffmpeg`).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import isaacsim  # noqa: E402
from isaacsim import SimulationApp  # noqa: E402

# (azimuth about +Z [deg], elevation [deg], distance as a multiple of the part's diagonal,
#  aim-point lift as a multiple of that diagonal, track the disc, hide the disc)
VIEWS = {
    "iso": (215.0, 34.0, 1.15, 0.0, False, False),
    "across": (270.0, 11.0, 1.05, 0.02, False, False),
    "along": (185.0, 13.0, 1.10, 0.02, False, False),
    "top": (270.0, 84.0, 1.10, 0.0, False, True),
    "chase": (215.0, 26.0, 0.55, 0.01, True, False),
}

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("runs", type=Path, nargs="+", help="run directories holding press.usda")
parser.add_argument("--out", type=Path, default=Path("runs/part_videos"))
parser.add_argument("--views", nargs="+", default=list(VIEWS), choices=list(VIEWS))
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--fps", type=float, default=6.0, help="playback rate of the mp4")
parser.add_argument("--hold", type=int, default=1, help="repeat each press this many video frames")
parser.add_argument("--settle", type=int, default=4,
                    help="render updates per frame before capturing; raise if frames look noisy")
parser.add_argument("--focal-length", type=float, default=18.0, help="camera focal length [mm]")
parser.add_argument("--keep-frames", action="store_true", help="keep the intermediate PNGs")
args = parser.parse_args()

recordings = []
for run in args.runs:
    usd = run if run.suffix in (".usd", ".usda") else run / "press.usda"
    if usd.exists():
        recordings.append((run.name if run.is_dir() else usd.parent.name, usd.resolve()))
    else:
        print(f"{run}: no press.usda, skipped")
if not recordings:
    raise SystemExit("nothing to render")

experience = os.path.join(os.path.dirname(isaacsim.__file__), "apps", "isaacsim.exp.full.newton.kit")
simulation_app = SimulationApp(
    {
        "headless": True,
        "width": args.width,
        "height": args.height,
        "extra_args": [
            "--/exts/isaacsim.core.throttling/enable_async=false",
            "--/app/asyncRendering=false",
            "--/app/asyncRenderingLowLatency=false",
            "--/isaac/startup/ros_bridge_extension=",
            "--/crashreporter/enabled=false",
            # Segfaults in _clear_overlay whenever the stage is swapped, which this does per run.
            "--/exts/isaacsim.robot.virtual_gantry/enable=false",
        ],
    },
    experience=experience,
)

import imageio.v2 as imageio  # noqa: E402
import omni.kit.viewport.utility as vp_utils  # noqa: E402
import omni.timeline  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
import carb  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

CAMERA_PATH = "/World/ReviewCamera"

# The grid, origin axes and selection outline are drawn over the render and end up in the capture.
# The viewport extension sets these itself at startup, so command-line overrides do not survive —
# they have to be turned off once the app is up.
_settings = carb.settings.get_settings()
for _key in ("/app/viewport/grid/enabled", "/app/viewport/outline/enabled",
             "/persistent/app/viewport/Viewport/Viewport0/guide/grid/visible",
             "/persistent/app/viewport/Viewport/Viewport0/guide/axis/visible",
             "/persistent/app/viewport/Viewport/Viewport0/guide/selection/visible"):
    _settings.set(_key, False)


def look_at(camera: UsdGeom.Camera, eye: np.ndarray, target: np.ndarray) -> None:
    """Point a USD camera (which looks down its own -Z) at a world-space target."""
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)),
                                   Gf.Vec3d(0.0, 0.0, 1.0))
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(view.GetInverse())


def eye_for(centre: np.ndarray, diagonal: float, azimuth: float, elevation: float,
            distance: float) -> np.ndarray:
    a, e = math.radians(azimuth), math.radians(elevation)
    radius = diagonal * distance
    return centre + radius * np.array([math.cos(a) * math.cos(e),
                                       math.sin(a) * math.cos(e),
                                       math.sin(e)])


def pad_translations(stage, start: float, end: float) -> dict[float, np.ndarray]:
    """Where the disc is at each time code, so the chase view can follow it."""
    pad = UsdGeom.Xformable(stage.GetPrimAtPath("/World/pad"))
    ops = [op for op in pad.GetOrderedXformOps()
           if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if not ops:
        return {}
    op = ops[0]
    return {t: np.array(op.Get(t), dtype=np.float64) for t in np.arange(start, end + 1.0)}


def render(name: str, usd: Path, frames_dir: Path) -> list[tuple[str, Path, int]]:
    stage_utils.open_stage(str(usd))
    simulation_app.update()
    stage = stage_utils.get_current_stage(backend="usd")
    # Kit writes viewport state into customLayerData on shutdown; a recording once lost 69 of its
    # 72 frames that way. Rendering is strictly read-only.
    stage.GetRootLayer().SetPermissionToSave(False)

    start, end = float(stage.GetStartTimeCode()), float(stage.GetEndTimeCode())
    fps = stage.GetTimeCodesPerSecond() or 12.0
    bounds = UsdGeom.BBoxCache(start, [UsdGeom.Tokens.default_]).ComputeWorldBound(
        stage.GetPrimAtPath("/World/sheet")).ComputeAlignedRange()
    lo = np.array(bounds.GetMin(), dtype=np.float64)
    hi = np.array(bounds.GetMax(), dtype=np.float64)
    centre, diagonal = (lo + hi) / 2.0, float(np.linalg.norm(hi - lo))
    pads = pad_translations(stage, start, end)

    camera = UsdGeom.Camera.Define(stage, CAMERA_PATH)
    camera.CreateFocalLengthAttr(args.focal_length)  # 18 mm ~= 60 deg on the default aperture
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    viewport = vp_utils.get_active_viewport()
    viewport.resolution = (args.width, args.height)
    viewport.camera_path = CAMERA_PATH
    timeline = omni.timeline.get_timeline_interface()

    produced = []
    for view in args.views:
        azimuth, elevation, distance, lift, chase, hide_pad = VIEWS[view]
        # The plan view exists to compare the patch against the green band, so the disc — even at
        # 40% opacity — is just in the way.
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/pad")).GetVisibilityAttr().Set(
            UsdGeom.Tokens.invisible if hide_pad else UsdGeom.Tokens.inherited)
        view_dir = frames_dir / f"{name}_{view}"
        view_dir.mkdir(parents=True, exist_ok=True)
        began = time.perf_counter()
        for step, code in enumerate(np.arange(start, end + 1.0)):
            aim = centre.copy()
            if chase and code in pads:
                aim = pads[code].copy()
                aim[2] = centre[2]
            aim[2] += diagonal * lift
            look_at(camera, eye_for(aim, diagonal, azimuth, elevation, distance), aim)
            timeline.set_current_time(float(code) / fps)
            for _ in range(args.settle):
                simulation_app.update()
            vp_utils.capture_viewport_to_file(viewport, str(view_dir / f"f{step:04d}.png"))
            for _ in range(2):          # let the capture flush before the next time code
                simulation_app.update()
        produced.append((view, view_dir, int(end - start) + 1))
        print(f"  {name}/{view}: {int(end-start)+1} frames in {time.perf_counter()-began:.0f}s",
              flush=True)
    return produced


def encode(name: str, view: str, view_dir: Path, out: Path) -> Path | None:
    pngs = sorted(view_dir.glob("f*.png"))
    if not pngs:
        print(f"  {name}/{view}: no frames captured")
        return None
    path = out / f"{name}_{view}.mp4"
    writer = imageio.get_writer(path, fps=args.fps, codec="libx264", quality=7,
                                macro_block_size=None, ffmpeg_params=["-pix_fmt", "yuv420p"])
    for png in pngs:
        frame = imageio.imread(png)[:, :, :3]
        for _ in range(max(1, args.hold)):
            writer.append_data(frame)
    writer.close()
    return path


def main() -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    frames_root = args.out / "_frames"
    began = time.perf_counter()
    written = []
    for name, usd in recordings:
        print(f"{name}: {usd}", flush=True)
        for view, view_dir, _ in render(name, usd, frames_root):
            path = encode(name, view, view_dir, args.out)
            if path:
                written.append(path)
    if not args.keep_frames and frames_root.exists():
        for png in frames_root.rglob("*.png"):
            png.unlink()
        for directory in sorted(frames_root.glob("*"), reverse=True):
            directory.rmdir()
        frames_root.rmdir()

    print(f"\n{len(written)} videos in {args.out} ({time.perf_counter()-began:.0f}s)")
    for path in written:
        print(f"  {path}  {path.stat().st_size/1e6:.1f} MB")
    return 0


try:
    code = main()
finally:
    simulation_app.close()
raise SystemExit(code)
