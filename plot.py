import matplotlib.pyplot as plt
import numpy as np
import os

file = 'propeller_angles_results_TripleCAg2T.txt'

angles_by_pair = {}

with open('results/' + file, 'r') as f:
    lines = f.readlines()

# parse header and data
for line in lines[1:]:
    if line.strip():
        parts = line.strip().split()
        if len(parts) >= 5:
            pair_index = int(parts[1])  # column 2 is pair number (1-indexed)
            angle = float(parts[4])     # column 5 is the angle

            if pair_index not in angles_by_pair:
                angles_by_pair[pair_index] = []
            angles_by_pair[pair_index].append(angle)

num_frames = len(next(iter(angles_by_pair.values())))

time_ns = np.array(range(num_frames)) * 100.0 / num_frames

name_part = file.removeprefix('propeller_angles_results_').removesuffix('.txt')
output_folder = f'results/{name_part}_plots'
os.makedirs(output_folder, exist_ok=True)

# creates the average line
def moving_average(data, window_size=2750): # increasing window_size increases 'smoothness'
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# plot for each base pair
for pair_index, angles in angles_by_pair.items():
    plt.figure(figsize=(8, 4))
    plt.plot(time_ns, angles, color='#0265a3', linewidth=0.6, alpha=0.8)

    #  smoothed line
    smooth_angles = moving_average(angles)
    plt.plot(time_ns, smooth_angles, color='black', linewidth=1.5)

    plt.xlabel('Time (ns)', fontsize=14)
    plt.ylabel('Propeller Angle (degrees)', fontsize=14)

    plt.xticks(ticks=range(0, 101, 20), fontsize=12)
    plt.tick_params(axis='x', which='both', direction='out', length=4, bottom=True, top=False) 

    plt.title(f'Base Pair #{pair_index} - Propeller Angle vs Time (1000 frames/ns)')
    plt.yticks(ticks=range(-90, 91, 45), fontsize=12)
    plt.ylim(-90, 90)
    plt.grid(axis='y', alpha=0.3) 
    plt.xlim(-5, 105)
    plt.tight_layout()

    filename = f'{name_part}_basepair#{pair_index}.jpg'
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {filepath}")