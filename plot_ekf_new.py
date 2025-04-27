import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from numpy import unwrap
from matplotlib.ticker import AutoMinorLocator

data = np.load('ekf_data_plot.npz')

# Set global font sizes for all plots
plt.rcParams.update({
    'font.size': 18,              # Base font size
    'axes.titlesize': 32,         # Title size
    'axes.labelsize': 20,         # Axis label size
    'xtick.labelsize': 18,        # X tick label size
    'ytick.labelsize': 18,        # Y tick label size
    'legend.fontsize': 20,        # Legend font size
    'figure.titlesize': 20,       # Figure title size
    'figure.figsize': (10, 15),   # Larger figure
    'lines.linewidth': 2,         # Thicker lines
    'lines.markersize': 8         # Larger markers
})

start_index = 10

# Normalize time to start at 0
normalized_time = data['time'][start_index:] - data['time'][start_index]


def create_confidence_band(x, y, covs, confidence=2):
    """Create confidence bands along the trajectory path"""
    upper_band = []
    lower_band = []

    for i in range(len(x)):
        # Calculate heading angle from trajectory
        if i > 0 and i < len(x)-1:
            # Use both previous and next points for smoother heading
            dx = x[i+1] - x[i-1]
            dy = y[i+1] - y[i-1]
        elif i > 0:
            # End point - use previous point
            dx = x[i] - x[i-1]
            dy = y[i] - y[i-1]
        else:
            # Start point - use next point
            dx = x[1] - x[0]
            dy = y[1] - y[0]

        heading = np.arctan2(dy, dx)

        # Direction perpendicular to trajectory
        perp_x = -np.sin(heading)
        perp_y = np.cos(heading)

        # Extract the position covariance
        pos_cov = covs[i][:2, :2]

        # Project covariance onto perpendicular direction
        perp_dir = np.array([perp_x, perp_y])
        variance_along_perp = perp_dir.T @ pos_cov @ perp_dir
        std_dev = np.sqrt(variance_along_perp)

        # Add points perpendicular to path
        upper_band.append((x[i] + confidence * std_dev * perp_x,
                          y[i] + confidence * std_dev * perp_y))
        lower_band.append((x[i] - confidence * std_dev * perp_x,
                          y[i] - confidence * std_dev * perp_y))

    return upper_band, lower_band


# Create new figures with better formatting - explicitly don't share x axis
fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=False)
# fig.suptitle("Extended Kalman Filter Performance", fontsize=16)

# Position plot (x-axis is x position, y-axis is y position)
ax_pos = axes[0]
ax_pos.set_title("Vehicle Position Tracking", fontsize=20)
ax_pos.set_xlabel("X Position (m)")  # Set proper label for position plot
ax_pos.set_ylabel("Y Position (m)")

# Ground truth trajectory
ax_pos.plot(data['gt_position'][start_index:, 0], data['gt_position'][start_index:, 1],
            'b-', linewidth=2, label="Ground Truth")

# GPS measurements
ax_pos.plot(data['gps_position'][start_index:, 0], data['gps_position'][start_index:, 1],
            'g.', alpha=0.4, label="GPS Readings")

# EKF estimates
ax_pos.plot(data['ekf_position'][start_index:, 0], data['ekf_position'][start_index:, 1],
            'r-', linewidth=1.5, label="EKF Estimate")

# Create continuous error bounds
x_pos = data['ekf_position'][start_index:, 0]
y_pos = data['ekf_position'][start_index:, 1]

# Calculate upper and lower bounds
confidence = 2  # 2-sigma (95% confidence)

# Generate the confidence bands
upper_band, lower_band = create_confidence_band(
    x_pos, y_pos, data['ekf_covariance'][start_index:], confidence=confidence)
# Convert to numpy arrays for easier plotting
upper_band = np.array(upper_band)
lower_band = np.array(lower_band)

# Plot confidence bands
ax_pos.plot(upper_band[:, 0], upper_band[:, 1], 'y-', alpha=0.5, linewidth=1)
ax_pos.plot(lower_band[:, 0], lower_band[:, 1], 'y-', alpha=0.5, linewidth=1)

# Fill between the bands
band_x = np.concatenate([upper_band[:, 0], lower_band[::-1, 0]])
band_y = np.concatenate([upper_band[:, 1], lower_band[::-1, 1]])
ax_pos.fill(band_x, band_y, color='yellow', alpha=0.2, label='95% Confidence')

# Make position plot aspect ratio equal so circles look like circles
ax_pos.set_aspect('equal')

# Yaw angle plot (x-axis is time)
ax_yaw = axes[1]
ax_yaw.set_title("Yaw Angle Tracking", fontsize=20)
ax_yaw.set_xlabel("Time (s)")  # Set proper label
ax_yaw.set_ylabel("Yaw (rad)")

# Unwrap the angles to remove discontinuities
unwrapped_gt_yaw = unwrap(data['gt_yaw'][start_index:])
unwrapped_ekf_yaw = unwrap(data['ekf_yaw'][start_index:])

# Plot unwrapped angles using normalized time
ax_yaw.plot(normalized_time, unwrapped_gt_yaw, 'b-', linewidth=2, label="Ground Truth")
ax_yaw.plot(normalized_time, unwrapped_ekf_yaw, 'r-', linewidth=1.5, label="EKF Estimate")

# Extract yaw standard deviation
yaw_std = np.array([np.sqrt(cov[2, 2]) for cov in data['ekf_covariance'][start_index:]])

# Use unwrapped angles for confidence bounds
yaw_upper = unwrapped_ekf_yaw + confidence * yaw_std
yaw_lower = unwrapped_ekf_yaw - confidence * yaw_std

# Plot error bounds for yaw using normalized time
ax_yaw.fill_between(normalized_time, yaw_lower, yaw_upper, color='yellow', alpha=0.2,
                    label='95% Confidence')

# Velocity plot (x-axis is time)
ax_vel = axes[2]
ax_vel.set_title("Velocity Tracking", fontsize=20)
ax_vel.set_xlabel("Time (s)")
ax_vel.set_ylabel("Velocity (km/h)")

# Convert velocity from m/s to km/h (multiply by 3.6)
vel_kmh = data['velocity'][start_index:] * 3.6
gt_vel_kmh = data['gt_velocity'][start_index:] * 3.6

# Plot velocity in km/h using normalized time
ax_vel.plot(normalized_time, vel_kmh, 'r-', linewidth=1.5, label='EKF Velocity')
ax_vel.plot(normalized_time, gt_vel_kmh, 'b-', linewidth=1.5, label='GT Velocity')
# Extract velocity standard deviation
vel_std = np.array([np.sqrt(cov[3, 3]) for cov in data['ekf_covariance'][start_index:]])
vel_upper_kmh = vel_kmh + confidence * vel_std * 3.6  # Convert std to km/h too
vel_lower_kmh = vel_kmh - confidence * vel_std * 3.6

# Plot error bounds for velocity using normalized time
ax_vel.fill_between(normalized_time, vel_lower_kmh, vel_upper_kmh,
                   color='yellow', alpha=0.2, label='95% Confidence')

# Add better x-axis ticks
for ax in axes:
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.legend(loc='best', fontsize=16, framealpha=0.8)


# Set time axis limits for yaw and velocity plots only
time_min = 0
time_max = normalized_time[-1]
ax_yaw.set_xlim(time_min, time_max)
ax_vel.set_xlim(time_min, time_max)

# Make sure position plot has appropriate axis limits
ax_pos.set_xlim(np.min(x_pos) - 5, np.max(x_pos) + 5)
ax_pos.set_ylim(np.min(y_pos) - 5, np.max(y_pos) + 5)

plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.45)  # Add more space between plots
plt.show()

# Optionally save the figure
plt.savefig('ekf_performance.png', dpi=300, bbox_inches='tight')