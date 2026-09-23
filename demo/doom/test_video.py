import time
from unittest.mock import Mock

import imageio_ffmpeg
import numpy as np
import pytest

from video import VideoRecorder


def test_timing_repeats_previous_frame_and_caps_duration(tmp_path):
    recorder = VideoRecorder(tmp_path / 'clip.mp4', seconds=30, fps=20)
    recorder.process = Mock()
    recorder.previous = b'old frame'
    recorder._flush(0.15)
    assert recorder.count == 3
    recorder._flush(0.15)
    assert recorder.count == 3
    recorder._flush(31)
    assert recorder.count == 600
    assert recorder.process.stdin.write.call_count == 600
    recorder.process.stdin.write.assert_called_with(b'old frame')


def test_mp4_encodes_and_decodes(tmp_path):
    path = tmp_path / 'clip.mp4'
    recorder = VideoRecorder(path, seconds=0.1)
    recorder.draw(np.zeros((120,160,3), dtype=np.uint8), ['Decision: attack', '120 ms'])
    time.sleep(0.12)
    recorder.close()
    frames = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
    try:
        metadata = next(frames)
        assert metadata['size'] == (1280,720)
        assert metadata['duration'] == pytest.approx(0.1, abs=0.01)
        assert len(list(frames)) == 2
    finally:
        frames.close()
    with pytest.raises(FileExistsError):
        VideoRecorder(path)
