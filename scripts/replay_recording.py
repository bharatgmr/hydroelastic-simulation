"""Replay a baked nut-and-bolt recording (see nut_bolt_hydroelastic.py --record).

Playback is pure USD animation: the timeline is scrubbed frame by frame and physics never runs, so
startup is seconds rather than minutes and every replay looks identical. Run:
    source env.sh
    python scripts/replay_recording.py recordings/nut_bolt.usda [--loop] [--speed 0.5]
"""

import argparse
import os
import time

import isaacsim
from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("recording", help="USD file written by nut_bolt_hydroelastic.py --record")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--loop", action="store_true", help="restart when the recording ends")
parser.add_argument("--speed", type=float, default=1.0, help="playback rate (0.5 = half speed)")
parser.add_argument("--eye", type=float, nargs=3, default=[0.12, 0.12, 0.08],
                    help="camera position [m]; the grind scenes are ~4x bigger than the nut/bolt")
parser.add_argument("--target", type=float, nargs=3, default=[0.0, 0.0, 0.02],
                    help="camera aim point [m]")
args = parser.parse_args()

recording = os.path.abspath(args.recording)
if not os.path.exists(recording):
    raise SystemExit(f"No such recording: {recording}")

experience = os.path.join(os.path.dirname(isaacsim.__file__), "apps", "isaacsim.exp.full.newton.kit")
simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "extra_args": [
            "--/exts/isaacsim.core.throttling/enable_async=false",
            "--/app/asyncRendering=false",
            "--/app/asyncRenderingLowLatency=false",
            "--/isaac/startup/ros_bridge_extension=",
            "--/crashreporter/enabled=false",
        ],
    },
    experience=experience,
)

import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
import omni.timeline  # noqa: E402

stage_utils.open_stage(recording)
simulation_app.update()

stage = stage_utils.get_current_stage(backend="usd")
# Replay is strictly read-only. Kit writes viewport camera state into the layer's customLayerData,
# which makes the stage dirty and gets saved back on shutdown — that silently truncated a 72-frame
# recording to the 3 frames a previous session had open. Refuse the save instead.
stage.GetRootLayer().SetPermissionToSave(False)
start, end = stage.GetStartTimeCode(), stage.GetEndTimeCode()
fps = stage.GetTimeCodesPerSecond() or 60.0
print(f"replaying {recording}: frames {start:.0f}-{end:.0f} at {fps:.0f} fps")

if not args.headless:
    import omni.kit.actions.core
    from isaacsim.core.rendering_manager import ViewportManager

    omni.kit.actions.core.get_action_registry().get_action(
        "omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera"
    ).execute()
    ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=args.eye, target=args.target)

# Set the time directly instead of timeline.play(): play would also start the physics engine.
#
# Render as fast as the app will go and advance the animation from the wall clock, rather than
# one update() per animation frame. Kit only processes window events inside update(), so on a
# 1 fps recording the one-update-per-frame version left the UI frozen for a second at a time and
# the desktop declared it "Not Responding".
timeline = omni.timeline.get_timeline_interface()
frame = float(start)
last = time.perf_counter()
while simulation_app.is_running():
    now = time.perf_counter()
    frame += (now - last) * fps * max(args.speed, 1e-3)
    last = now
    if frame > end:
        if not args.loop:
            break
        frame = float(start)
    timeline.set_current_time(frame / fps)
    simulation_app.update()

simulation_app.close()
