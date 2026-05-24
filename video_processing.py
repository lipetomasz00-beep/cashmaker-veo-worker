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

    subprocess.run(cmd, check=True)

    return output_path
