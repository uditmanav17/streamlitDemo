import io
import shutil as sh
import tempfile
import time
from pathlib import Path

import gdown as g
import numpy as np
import streamlit as st
import supervision as sv
import torch
from torchvision.ops import box_iou
from ultralytics import YOLO

DEMO_VIDEO_PATH = "./test_video.mp4"  # fmt: skip
DEMO_VIDEO_NAME = "test.mp4"
MODEL_PATH = "./yolo8_model.pt"

st.set_page_config(
    page_title="NFL Players Collision Detections",
    page_icon="🏈",
    # layout = 'wide'
)


@st.cache_resource
def load_model(model_path: str | Path):
    model = YOLO(model=model_path)
    return model


def main():
    st.title("Object Detection API")
    st.sidebar.title("Settings")

    # sidebar layout
    st.sidebar.markdown("---")
    confidence = st.sidebar.slider(
        "Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
    )
    frame_pause = st.sidebar.slider(
        "Pause frame when Collision (seconds)",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.2,
    )
    st.sidebar.markdown("---")

    # check if GPU is available
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        st.info("GPU available 🔥 - Predictions will be sped up")
    else:
        st.warning("GPU NOT available 🚨 - Predictions might be slow")

    # upload video
    video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=["mp4"])
    vid_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    print(vid_file.name)
    if video_file_buffer:
        file_bytes = io.BytesIO(video_file_buffer.read())
        with open(vid_file.name, "wb") as f:
            f.write(file_bytes.read())
    else:
        sh.copy(DEMO_VIDEO_PATH, vid_file.name)

    demo_bytes = vid_file.read()
    st.sidebar.text("Input Video")
    st.sidebar.video(demo_bytes)
    st.sidebar.markdown(
        "[More Video Samples](https://drive.google.com/drive/folders/1Dcnht4zeJWJzHg9Ev6BRsrJ85wKoYE7a)"
    )

    model = load_model(MODEL_PATH)
    # custom tracking frame by frame
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

    detection_button = st.button("Start Detections!")
    st_frame = st.empty()
    metric_placeholder = st.empty()

    total_collision_counts = 0
    if detection_button:
        for r in model.track(source=vid_file.name, stream=True, show=False):
            frame = r.orig_img
            detections = sv.Detections.from_ultralytics(r)
            if r.boxes.id is not None:
                detections.tracker_id = r.boxes.id.cpu().numpy().astype(int)
            # Filter detections based on class_id
            # detections = detections[detections.class_id != 0]
            # Filter detections based on confidence
            detections = detections[detections.confidence >= confidence]
            bboxes = detections.xyxy
            ious = box_iou(torch.Tensor(bboxes), torch.Tensor(bboxes)).numpy()
            filtered_ious_mask = (ious > 0) & (ious != 1)
            filtered_detections = np.where(filtered_ious_mask)
            collisions = list(r for r in filtered_detections[0])
            current_frame_collisions = len(collisions)

            new_class_ids = np.zeros_like(detections.class_id)
            new_class_ids[collisions] = 1

            detections.class_id = new_class_ids
            labels = []
            for idx, (bboxes, mask, conf, class_id, tracker_id) in enumerate(detections):  # fmt: skip
                if idx in collisions:
                    labels.append(f"P{tracker_id} {'BANG!'} {conf:0.2f}")  # fmt: skip
                else:
                    labels.append(f"P{tracker_id} {conf:0.2f}")  # fmt: skip

            frame = box_annotator.annotate(
                scene=frame,
                detections=detections,
                labels=labels,
            )
            st_frame.image(frame, channels="BGR")
            total_collision_counts += current_frame_collisions
            with metric_placeholder.container():
                col1, col2 = st.columns(2)
                col1.metric(f"Current Frame Collisions", current_frame_collisions)
                col2.metric(f"Total Collisions", total_collision_counts)
            if len(collisions) > 0:
                time.sleep(frame_pause)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    if not Path(MODEL_PATH).exists():
        drive_model_id = "1oQbFgQAxUM13m1nMmZ40QKNWho4Vllqq"
        g.download(id=drive_model_id, output=MODEL_PATH)
    if not Path(DEMO_VIDEO_PATH).exists():
        drive_video_id = "1qrbGE0-8OJAHHWIeiCekQ1QpzYfscUCL"
        g.download(id=drive_video_id, output=DEMO_VIDEO_PATH)

    main()
