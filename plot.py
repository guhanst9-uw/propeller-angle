import matplotlib.pyplot as plt
import numpy as np

file = 'propeller_angles_results_CHgT.txt'

angles = []
with open('results/'+file, 'r') as f:
    lines = f.readlines()
    
# skip header line and parse data
for line in lines[1:]:  # skip header line
    if line.strip():  # skip empty lines
        parts = line.strip().split()
        if len(parts) >= 5:
            angle = float(parts[4])  # propeller_Angle is the fifth column
            angles.append(angle)

print(f"Total data points: {len(angles)}")

# group by frames (7 angles per frame)
angles_per_frame = 7
num_frames = len(angles) // angles_per_frame
print(f"Number of frames: {num_frames}")

# calculate average angle per frame
frame_angles = []
for i in range(num_frames):
    start_idx = i * angles_per_frame
    end_idx = start_idx + angles_per_frame
    frame_avg = np.mean(angles[start_idx:end_idx])
    frame_angles.append(frame_avg)

# convert frames to time in nanoseconds (1000 frames per ns for 100ns total)
time_ns = np.array(range(num_frames)) * 100.0 / num_frames

plt.figure(figsize=(12, 6))
plt.plot(time_ns, frame_angles, 'b-', linewidth=0.5, alpha=0.8)

# smoothing
window_size = 2000  # smooth over 20ns windows
if len(frame_angles) > window_size:
    smoothed = np.convolve(frame_angles, np.ones(window_size)/window_size, mode='valid')
    smoothed_time = time_ns[window_size//2:-window_size//2+1]
    plt.plot(smoothed_time, smoothed, 'darkblue', linewidth=2, label='Smoothed')

plt.xlabel('Time (ns)', fontsize=14)
plt.ylabel('Propeller Angle (degrees)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlim(0, 100)

y_min, y_max = min(frame_angles), max(frame_angles)
y_range = y_max - y_min
plt.ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)

plt.title(f'Propeller Angle vs Time ({num_frames} frames, 100 ns total) for ' + file.rsplit('_', 1)[-1].removesuffix('.txt'))
if len(frame_angles) > window_size:
    plt.legend()

plt.tight_layout()

plt.savefig('results/propeller_angle_plot_CHgT.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'results/propeller_angle_plot_CHgT.png'")
plt.show()