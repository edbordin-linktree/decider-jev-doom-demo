"""Wall-clock MP4 recording of game frames and the decision panel."""
import math
from pathlib import Path
import subprocess
import time

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont


class VideoRecorder:
    def __init__(self, path, seconds=30, label="Decider / ExecuTorch MLX", fps=20):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(f"Refusing to overwrite {self.path}")
        if seconds <= 0:
            raise ValueError("Video duration must be positive")
        self.seconds, self.label, self.fps = seconds, label, fps
        self.started = None
        self.count = 0
        self.previous = None
        self.process = None
        try:
            self.font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
        except OSError:
            self.font = ImageFont.load_default(size=14)

    @property
    def elapsed(self):
        return 0 if self.started is None else time.monotonic() - self.started

    def _flush(self, elapsed):
        target = math.ceil(min(elapsed, self.seconds) * self.fps)
        while self.previous is not None and self.count < target:
            self.process.stdin.write(self.previous)
            self.count += 1

    def draw(self, frame, panel):
        canvas = Image.new("RGB", (1280, 720), "#16162b")
        canvas.paste(Image.fromarray(frame).resize((720, 540), Image.Resampling.NEAREST), (0, 90))
        draw = ImageDraw.Draw(canvas)
        draw.text((18, 18), self.label, font=self.font, fill="#f6d8aa")
        draw.text((18, 44), "Local inference | wall-clock playback | no audio", font=self.font, fill="#b6b6ca")
        draw.text((18, 658), "Dasein Labs harness / Mapika Decider / ExecuTorch MLX", font=self.font, fill="#b6b6ca")
        for index, line in enumerate(panel[:42]):
            draw.text((738, 10 + index * 16), line, font=self.font, fill="#f6d8aa")
        if self.started is None:
            self.process = subprocess.Popen([
                imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-n",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720", "-r", str(self.fps),
                "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "23", "-threads", "2", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(self.path),
            ], stdin=subprocess.PIPE, start_new_session=True)
            self.started = time.monotonic()
        self._flush(self.elapsed)
        self.previous = canvas.tobytes()

    def close(self):
        if self.process is None:
            return
        try:
            self._flush(self.elapsed)
        finally:
            self.process.stdin.close()
            try:
                code = self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                raise RuntimeError("Video encoder did not finish") from None
        if code:
            raise RuntimeError(f"Video encoder exited with status {code}")
