import os
import subprocess

def extract_frames_from_folders(folders, output_root_dir, fps=24):
    for folder in folders:
        # 遍历每个文件夹中的 mp4 文件
        for video_file in os.listdir(folder):
            if video_file.endswith('.mp4'):
                video_path = os.path.join(folder, video_file)
                video_name = os.path.splitext(video_file)[0]
                
                output_folder = os.path.join(output_root_dir, video_name)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                
                # 使用 ffmpeg 提取帧
                cmd = [
                    "ffmpeg", "-i", video_path, "-vf", f"fps={fps}",
                    os.path.join(output_folder, "%04d.png")
                ]
                print(f"Extracting frames from {video_file}...")
                subprocess.call(cmd)

if __name__ == "__main__":
    # 定义视频文件夹路径
    video_folders = [
        "/data0/tw/MusePose_examples_new/MusePose_rendered",
        # "/data0/tw/MusePose_outputs/MusePose_rendered",
        # "/data0/tw/Mimicmotion_outputs"
    ]
    
    # 输出帧的根目录
    output_dir = '/data0/lfz/data/ACA/test_bgr'

    
    # 开始提取
    extract_frames_from_folders(video_folders, output_dir, fps=24)
