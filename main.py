import csv
import copy
import itertools
import threading
import urllib

import aiohttp
import uvicorn
import websocket
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import websockets
import json
import base64
from io import BytesIO
from PIL import Image

import cv2 as cv
import numpy as np
import mediapipe as mp
from model import KeyPointClassifier


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def draw_info_text(image, brect, facial_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    if facial_text != "":
        info_text = 'Emotion :' + facial_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image


# Initialize FastAPI
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FastAPI endpoints

cap_device = 0
cap_width = 1920
cap_height = 1080

use_brect = True

# Model load
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

keypoint_classifier = KeyPointClassifier()

# Read labels
with open('model/keypoint_classifier/keypoint_classifier_label.csv',
          encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]

mode = 0

# Global variables
connected_clients = set()
processing = True


class ESP32Camera:
    def __init__(self, ip_address):
        self.ip_address = ip_address
        self.ws = None
        self.connected = False
        self.retry_count = 0
        self.max_retries = 5
        self.face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    async def connect(self):
        while self.retry_count < self.max_retries:
            try:
                # Increase timeouts and disable SSL verification
                self.ws = await websockets.connect(
                    f'ws://{self.ip_address}:81',
                    ping_interval=None,  # Disable ping/pong
                    ping_timeout=None,
                    close_timeout=None,
                    max_size=None,  # No limit on message size
                    # read_limit=2 ** 20,  # Increase read buffer
                    write_limit=2 ** 20  # Increase write buffer
                )
                self.connected = True
                self.retry_count = 0
                print(f"Connected to ESP32-CAM at {self.ip_address}")
                return True
            except Exception as e:
                self.retry_count += 1
                print(f"Connection attempt {self.retry_count} failed: {str(e)}")
                await asyncio.sleep(2)  # Wait before retrying

        print("Max retry attempts reached")
        return False

    async def send_signal_to_esp32(self, emotion_data):
        """Send signal to ESP32 based on emotion data"""
        try:
            # Prepare the data to send to ESP32

            # ESP32 endpoint URL
            esp32_url = f"http://172.20.10.4/emotion"

            async with aiohttp.ClientSession() as session:
                async with session.post(esp32_url, json=emotion_data) as response:
                    if response.status == 200:
                        print("Successfully sent signal to ESP32")
                    else:
                        print(f"Failed to send signal to ESP32. Status: {response.status}")
        except Exception as e:
            print("Error sending signal to ESP32: {str(e)}")

    async def process_frame(self, frame):
        image = cv.flip(frame, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True

        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, face_landmarks)

                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, face_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)

                # emotion classification
                facial_emotion_id = keypoint_classifier(pre_processed_landmark_list)
                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    keypoint_classifier_labels[facial_emotion_id])

                # Update latest data
                global latest_data

                latest_data = mapEmotion(keypoint_classifier_labels[facial_emotion_id])
                await self.send_signal_to_esp32(latest_data)

        _, buffer = cv.imencode('.jpg', debug_image)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        return jpg_as_text

    async def receive_frames(self):
        while processing:
            if not self.connected:
                await self.connect()
                continue

            try:
                message = await self.ws.recv()

                # Convert blob data to numpy array
                image_data = BytesIO(message)
                pil_image = Image.open(image_data)
                frame = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)

                if frame is not None:
                    # Process frame
                    processed_frame = await self.process_frame(frame)
                    if processed_frame:
                        # Broadcast to all clients
                        await broadcast_frame(processed_frame)

            except websockets.exceptions.ConnectionClosed:
                print("Connection to ESP32-CAM closed")
                self.connected = False
                await asyncio.sleep(1)
            except Exception as e:
                print(f"Error: {str(e)}")
                self.connected = False
                await asyncio.sleep(1)


async def broadcast_frame(frame):
    if connected_clients:
        disconnected_clients = set()
        for client in connected_clients:
            try:
                await client.send(frame)
            except:
                disconnected_clients.add(client)

        # Remove disconnected clients
        connected_clients.difference_update(disconnected_clients)


async def handle_client(websocket):
    try:
        # Handle new client connection
        connected_clients.add(websocket)
        print(f"New client connected. Total clients: {len(connected_clients)}")

        # Keep the connection alive
        await websocket.wait_closed()
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.remove(websocket)
        print(f"Client disconnected. Total clients: {len(connected_clients)}")


async def main():
    # ESP32-CAM IP address
    esp32_ip = "172.20.10.2"  # Replace with your ESP32-CAM IP

    # Create ESP32 camera handler
    camera = ESP32Camera(esp32_ip)

    # Start WebSocket server for clients
    server = await websockets.serve(
        handle_client,
        "localhost",  # Listen on all interfaces
        8765,  # Port for clients to connect
        ping_interval=None,
        ping_timeout=None,
        close_timeout=None
    )

    print("WebSocket server started on ws://0.0.0.0:8765")

    # Start camera frame processing
    await asyncio.gather(
        camera.receive_frames(),
        server.wait_closed()
    )


@app.get("/emotions")
async def get_emotions():
    """Get only the latest emotion data"""
    return latest_data


def run_fastapi():
    """Run FastAPI server"""
    uvicorn.run(app, host="0.0.0.0", port=8000)


def mapEmotion(emotion):
    if emotion == "Happy":
        return "forward"
    elif emotion == "Angry":
        return "backward"
    elif emotion == "Neutral":
        return "stop"
    elif emotion == "Surprise":
        return "left"
    elif emotion == "Sad":
        return "right"


if __name__ == "__main__":
    try:
        # Start FastAPI in a separate thread
        fastapi_thread = threading.Thread(target=run_fastapi)
        fastapi_thread.start()

        # Run WebSocket server in main thread
        asyncio.run(main())
    except KeyboardInterrupt:
        processing = False
        print("\nShutting down...")
