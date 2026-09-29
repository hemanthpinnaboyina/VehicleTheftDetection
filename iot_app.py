
import streamlit as st
import cv2
import time

from ultralytics import YOLO
from iot_camera import IoTCamera, detect_vehicles


st.set_page_config(
    page_title="IoT Vehicle Detection",
    page_icon="🚗",
    layout="wide"
)


st.title("🚗 Vehicle Theft Detection and Localization")

st.subheader("IoT Camera + YOLOv8 Deep Learning")

st.info(
    "Prototype: Laptop webcam is used as a simulated IoT camera."
)


st.markdown("### Camera Input")


confidence = st.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.35,
    step=0.05
)


# Session state

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

if "total_frames" not in st.session_state:
    st.session_state.total_frames = 0

if "total_detections" not in st.session_state:
    st.session_state.total_detections = 0

if "detected_vehicle_types" not in st.session_state:
    st.session_state.detected_vehicle_types = set()

if "highest_confidence" not in st.session_state:
    st.session_state.highest_confidence = 0.0


col1, col2 = st.columns(2)


with col1:

    start_camera = st.button(
        "📷 Start Camera Detection",
        use_container_width=True
    )


with col2:

    stop_camera = st.button(
        "⏹️ Stop Camera",
        use_container_width=True
    )


# Start camera

if start_camera:

    st.session_state.camera_running = True

    st.session_state.total_frames = 0

    st.session_state.total_detections = 0

    st.session_state.detected_vehicle_types = set()

    st.session_state.highest_confidence = 0.0


# Stop camera

if stop_camera:

    st.session_state.camera_running = False

    st.success("Camera stopped successfully.")


# Camera processing

if st.session_state.camera_running:

    st.write("Loading YOLOv8 model...")

    model = YOLO("yolov8n.pt")

    camera = IoTCamera(camera_index=0)

    frame_placeholder = st.empty()

    current_placeholder = st.empty()

    summary_placeholder = st.empty()

    try:

        camera.start()

        st.success("Camera connected successfully.")

        # Process a limited number of frames per session.
        # This avoids an endless blocking loop.

        max_frames = 100

        for frame_number in range(max_frames):

            success, frame = camera.read_frame()

            if not success:

                st.error("Unable to read camera frame.")

                break

            annotated_frame, detected_vehicles = detect_vehicles(
                model,
                frame,
                confidence
            )

            st.session_state.total_frames += 1

            st.session_state.total_detections += len(
                detected_vehicles
            )

            for vehicle in detected_vehicles:

                vehicle_name = vehicle["name"]

                vehicle_confidence = vehicle["confidence"]

                st.session_state.detected_vehicle_types.add(
                    vehicle_name
                )

                if vehicle_confidence > st.session_state.highest_confidence:

                    st.session_state.highest_confidence = (
                        vehicle_confidence
                    )

            frame_rgb = cv2.cvtColor(
                annotated_frame,
                cv2.COLOR_BGR2RGB
            )

            frame_placeholder.image(
                frame_rgb,
                caption="Live IoT Camera Detection",
                use_container_width=True
            )

            # Current frame results

            current_placeholder.empty()

            with current_placeholder.container():

                st.markdown("### 🔍 Current Frame Results")

                if detected_vehicles:

                    for index, vehicle in enumerate(
                        detected_vehicles,
                        start=1
                    ):

                        st.write(
                            f"🚗 Vehicle {index}: "
                            f"{vehicle['name'].upper()} | "
                            f"Confidence: "
                            f"{vehicle['confidence']:.2f}"
                        )

                    st.success(
                        f"Vehicles detected in current frame: "
                        f"{len(detected_vehicles)}"
                    )

                else:

                    st.warning(
                        "No vehicle found in current frame."
                    )

            time.sleep(0.03)

        # Stop after processing

        st.session_state.camera_running = False

    except Exception as error:

        st.error(f"Camera error: {error}")

        st.session_state.camera_running = False

    finally:

        camera.stop()


# Final detection summary

if not st.session_state.camera_running:

    if st.session_state.total_frames > 0:

        st.divider()

        st.markdown("## 📊 Detection Summary")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Processed Frames",
                st.session_state.total_frames
            )

        with col2:

            st.metric(
                "Total Vehicle Detections",
                st.session_state.total_detections
            )

        with col3:

            st.metric(
                "Highest Confidence",
                f"{st.session_state.highest_confidence:.2f}"
            )

        st.markdown("### 🚗 Vehicle Types Detected")

        if st.session_state.detected_vehicle_types:

            st.write(
                ", ".join(
                    sorted(
                        st.session_state.detected_vehicle_types
                    )
                )
            )

        else:

            st.info("No vehicles detected.")

        st.info(
            "Note: This module detects vehicles in CCTV frames. "
            "It does not yet identify stolen vehicles or prove theft."
        )

    else:

        st.info(
            "Click 'Start Camera Detection' to begin."
        )