"""Shared deterministic H.264 encoding for offline visualization layers."""
from pathlib import Path
import imageio_ffmpeg


def open_video_writer(output, width=1920, height=1080, fps=30):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    writer=imageio_ffmpeg.write_frames(str(output),(width,height),fps=fps,codec='libx264',
        pix_fmt_out='yuv420p',quality=8,macro_block_size=1,output_params=['-movflags','+faststart'])
    writer.send(None)
    return writer
