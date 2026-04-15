#!/usr/bin/env python3
import socket
import struct
import cv2
import numpy as np
import time

# =============================================================================
# CONFIGURATION - Change this to your AI-deck's IP address!
# =============================================================================
ESP32_IP = "172.16.0.39"  # <-- PUT YOUR AI-DECK IP HERE
ESP32_PORT = 5000
UDP_PORT = 5001
MAGIC_BYTE = b'FER'

# =============================================================================
# Protocol constants
# =============================================================================
CPX_HEADER_SIZE = 4
IMG_HEADER_MAGIC = 0xBC
IMG_HEADER_SIZE = 11

# Create socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

sock.bind(("0.0.0.0", UDP_PORT))

print("=" * 60)
print("AI-deck UDP Video Stream Client (Station Mode)")
print("=" * 60)
print(f"Your computer must be on the same network as the AI-deck")
print(f"AI-deck IP: {ESP32_IP}")
print(f"Listening on port: {UDP_PORT}")
print("-" * 60)

# Send magic byte to initiate connection
print(f"Sending magic byte to {ESP32_IP}:{ESP32_PORT}...")
sock.sendto(MAGIC_BYTE, (ESP32_IP, ESP32_PORT))
print("Waiting for video frames... Press Ctrl+C to exit")
print("-" * 60)

# Stream state
buffer = bytearray()
expected_size = None
receiving = False
frame_count = 0
last_time = None

while True:
    data, addr = sock.recvfrom(4096)
    
    # Check for image header
    if len(data) >= CPX_HEADER_SIZE + 1 and data[CPX_HEADER_SIZE] == IMG_HEADER_MAGIC:
        payload = data[CPX_HEADER_SIZE:]
        if len(payload) >= IMG_HEADER_SIZE:
            magic, width, height, depth, fmt, size = struct.unpack('<BHHBBI', payload[:IMG_HEADER_SIZE])
            expected_size = size
            buffer = bytearray(payload[IMG_HEADER_SIZE:])
            receiving = True
    elif receiving:
        buffer.extend(data[CPX_HEADER_SIZE:])
    
    # Check if frame complete
    if expected_size and len(buffer) >= expected_size:
        frame_count += 1
        
        # Calculate FPS
        now = time.time()
        if last_time:
            fps = 1.0 / (now - last_time)
            if frame_count % 10 == 0:
                print(f"Frame {frame_count}, FPS: {fps:.1f}")
        last_time = now
        
        # Decode image
        try:
            np_data = np.frombuffer(buffer[:expected_size], np.uint8)
            decoded = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
            
            if decoded is not None:
                cv2.imshow("AI-deck Stream", decoded)
                cv2.waitKey(1)
            else:
                # Try raw grayscale if JPEG decode fails
                print(f"Frame {frame_count}: Trying raw decode...")
        except Exception as e:
            print(f"Error: {e}")
        
        receiving = False
        expected_size = None