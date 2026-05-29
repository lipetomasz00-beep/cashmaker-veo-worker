import os
import subprocess


def concatenate_videos(video_paths, output_path):
    """
    Łączy wiele plików mp4 w jeden film.
    """

    concat_file = "/tmp/concat.txt"

    with open(concat_file, "w") as f:
        for path in video_paths:
            f.write(f"file '{path}'\n")

    cmd = [
        "ffmpeg",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_file,
        "-c",
        "copy",
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Install it with: apt-get install ffmpeg")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr}")

    return output_path
