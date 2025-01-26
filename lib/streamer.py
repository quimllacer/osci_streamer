import pyaudio
import numpy as np
from collections import deque
from mesh_generator import generate_mesh
from effects import effects
from mesh_to_edges import mesh_to_edges
from edges_to_path import  construct_graph, find_eulerian_path, interpolate_path
import time

# Initialize PyAudio
p = pyaudio.PyAudio()

# Parameters
chunk_size = 512*3  # Number of frames per buffer or "chunk"
rate = 96000*2        # Sample rate (e.g., 96 kHz)
channels = 2        # Stereo output
buffer_duration = 3  # Buffer duration in seconds
buffer_size = (rate * channels) * buffer_duration  # Max buffer size in frames
min_buffer = buffer_size // 5

time_per_mesh = 3
animation_speed = 1.2

audio_buffer = deque()  # Buffer to hold generated audio data
last_output = np.zeros(shape=(1, chunk_size*channels), dtype=np.float32)
n = 0

# # Audio generation function
# def generate_audio_data(n, timer, time_per_mesh):
#     t = np.linspace(0, 1, chunk_size, endpoint = False)
#     mesh, mesh_name = generate_mesh(timer, time_per_mesh)
#     mesh = effects(mesh, n)
#     edges = mesh_to_edges(mesh)[1]
#     path = find_eulerian_path(edges)
#     audio_image = interpolate_path(mesh.vertices[:,:2], path, t)
#     # print(audio_image.astype(np.float32).flatten().tobytes())
#     return audio_image.astype(np.float32).flatten()

def timer_function(t, period=0.60):
    distance = min(t, period - t)
    if t >= period:
        return 1
    else:
        return np.exp(-distance**2 / 0.00002)  # Adjust the constant for sharpness

# Callback function to be passed to PyAudio
def callback(in_data, frame_count, time_info, status):
    required_samples = frame_count * channels
    global last_output
    if len(audio_buffer) < required_samples:
        # Not enough data in buffer, fill with silence (or handle underflow)
        # output_data = np.zeros(required_samples, dtype=np.float32)
        output_data = last_output
    else:
        # Extract the exact amount of data needed for this callback
        output_data = np.array([audio_buffer.popleft() for _ in range(required_samples)])
        last_output = output_data

    return (output_data.tobytes(), pyaudio.paContinue)

# Open a stream with the callback
stream = p.open(format=pyaudio.paFloat32,
                output_device_index = None,
                channels=channels,
                rate=rate,
                frames_per_buffer=chunk_size,
                output=True,
                stream_callback=callback,
                )

# Start the stream
stream.start_stream()

# Continuously generate and buffer audio data
try:
    start_time = time.time()
    timer = 0
    mesh_idx = 0
    frame_idx = 0
    load_buffer = True
    while stream.is_active():
        # Pause generation if buffer is full (uncomment if stream_callback is active)
        if len(audio_buffer) <= min_buffer:
            load_buffer = True
            print(f"Buffer {len(audio_buffer)} after pause...")
        if len(audio_buffer) >= buffer_size:
            load_buffer = False

        if load_buffer == True:

            # Mesh selector based on timer
            if mesh_idx == 0 or timer >= time_per_mesh:
                    print(f'Index {mesh_idx}')
                    mesh, mesh_name = generate_mesh(mesh_idx, timer)
                    mesh_idx += 1
                    start_time = time.time()
                    frame_idx = 0


                    # Generate one image frame composed of "chunk_size" samples"
                    t = np.linspace(0, 1, chunk_size, endpoint = False)
                    # projected_mesh, edges = mesh_to_edges(effects(mesh, 360*animation_progress*animation_speed))
                    path = find_eulerian_path(mesh.edges_unique, construct_graph(mesh.edges_unique))
            animation_progress = frame_idx/(time_per_mesh*(rate/chunk_size))
            audio_image = interpolate_path(effects(mesh, 360*animation_progress*animation_speed).vertices[:,:2], path, t)
            audio_image /= np.amax(audio_image)


            # Buffer the generated audio data
            audio_buffer.extend(audio_image.astype(np.float32).flatten()) # Each run generates an image (or "frame").

            # For testing (comment if stream callback is active)
            # stream.write(audio_image.astype(np.float32).flatten().tobytes())

            frame_idx += 1
            timer = round(time.time() - start_time, 3)


except Exception as e:
    print(e)
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()
