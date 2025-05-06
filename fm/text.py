# test_frame_consistency.py
from fc_module import FrameConsistency, load_video_frames

if __name__ == "__main__":
    video = "b.mp4"
    frames = load_video_frames(video)
    print(f"{len(frames)} ")

    fc = FrameConsistency(
        version="openai/clip-vit-base-patch32",
        device="cuda",      
        mini_bsz=32,        
        return_type="float" 
    )

    score = fc(frames, step=1)
    print(f"Frame consistency (step=1): {score:.4f}")
