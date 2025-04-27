import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from numpy import unwrap

data = np.load('ekf_data.npz')

start_index = 50


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


# Create new figures with better formatting
fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
fig.suptitle("Extended Kalman Filter Performance", fontsize=16)

# Position plot
ax_pos = axes[0]
ax_pos.set_title("Vehicle Position Tracking", fontsize=14)
ax_pos.set_xlabel("X Position (m)")
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

# Create continuous error bounds (1-sigma)
x_pos = data['ekf_position'][start_index:, 0]
y_pos = data['ekf_position'][start_index:, 1]
x_std = np.array([np.sqrt(cov[0, 0]) for cov in data['ekf_covariance'][start_index:]])
y_std = np.array([np.sqrt(cov[1, 1]) for cov in data['ekf_covariance'][start_index:]])

# Calculate upper and lower bounds
confidence = 2  # 2-sigma (95% confidence)
x_upper = x_pos + confidence * x_std
x_lower = x_pos - confidence * x_std
y_upper = y_pos + confidence * y_std
y_lower = y_pos - confidence * y_std

# # Plot error bounds for X position
# ax_pos.fill_between(x_pos, y_lower, y_upper, color='yellow', alpha=0.2,
#                     label='95% Confidence')
# Generate the confidence bands
upper_band, lower_band = create_confidence_band(
    x_pos, y_pos, data['ekf_covariance'], confidence=confidence)
# Convert to numpy arrays for easier plotting
upper_band = np.array(upper_band)
lower_band = np.array(lower_band)

# Plot upper confidence band
ax_pos.plot(upper_band[:, 0], upper_band[:, 1], 'y-', alpha=0.5, linewidth=1)
# Plot lower confidence band
ax_pos.plot(lower_band[:, 0], lower_band[:, 1], 'y-', alpha=0.5, linewidth=1)

# Fill between the bands
# We need to handle this differently - we'll create a polygon
band_x = np.concatenate([upper_band[:, 0], lower_band[::-1, 0]])
band_y = np.concatenate([upper_band[:, 1], lower_band[::-1, 1]])
ax_pos.fill(band_x, band_y, color='yellow', alpha=0.2, label='95% Confidence')

# Yaw angle plot
ax_yaw = axes[1]
ax_yaw.set_title("Yaw Angle Tracking", fontsize=14)
ax_yaw.set_ylabel("Yaw (rad)")
# ax_yaw.plot(data['time'], data['gt_yaw'], 'b-', linewidth=2, label="Ground Truth")
# ax_yaw.plot(data['time'], data['ekf_yaw'], 'r-', linewidth=1.5, label="EKF Estimate")
# Unwrap the angles to remove discontinuities
unwrapped_gt_yaw = unwrap(data['gt_yaw'][start_index:])
unwrapped_ekf_yaw = -unwrap(data['ekf_yaw'][start_index:])

# Plot unwrapped angles
ax_yaw.plot(data['time'][start_index:], unwrapped_gt_yaw, 'b-', linewidth=2, label="Ground Truth")
ax_yaw.plot(data['time'][start_index:], unwrapped_ekf_yaw, 'r-', linewidth=1.5, label="EKF Estimate")

# Extract yaw standard deviation
yaw_std = np.array([np.sqrt(cov[2, 2]) for cov in data['ekf_covariance'][start_index:]])

# Use unwrapped angles for confidence bounds
yaw_upper = unwrapped_ekf_yaw + confidence * yaw_std
yaw_lower = unwrapped_ekf_yaw - confidence * yaw_std

# Plot error bounds for yaw
ax_yaw.fill_between(data['time'][start_index:], yaw_lower, yaw_upper, color='yellow', alpha=0.2,
                    label='95% Confidence')

# # Extract yaw standard deviation
# yaw_std = np.array([np.sqrt(cov[2, 2]) for cov in data['ekf_covariance']])
# yaw_upper = data['ekf_yaw'] + confidence * yaw_std
# yaw_lower = data['ekf_yaw'] - confidence * yaw_std

# # Plot error bounds for yaw
# ax_yaw.fill_between(data['time'], yaw_lower, yaw_upper, color='yellow', alpha=0.2,
#                     label='95% Confidence')

# Velocity plot
ax_vel = axes[2]
ax_vel.set_title("Velocity Estimate", fontsize=14)
ax_vel.set_xlabel("Time (s)")
ax_vel.set_ylabel("Velocity (m/s)")
ax_vel.plot(data['time'][start_index:], data['velocity'][start_index:], 'r-', linewidth=1.5, label='EKF Velocity')

# Extract velocity standard deviation
vel_std = np.array([np.sqrt(cov[3, 3]) for cov in data['ekf_covariance']][start_index:])
vel_upper = data['velocity'][start_index:] + confidence * vel_std
vel_lower = data['velocity'][start_index:] - confidence * vel_std

# Plot error bounds for velocity
ax_vel.fill_between(data['time'][start_index:], vel_lower, vel_upper, color='yellow', alpha=0.2,
                    label='95% Confidence')

# Add legends
for ax in axes:
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()

# Optionally save the figure
plt.savefig('ekf_performance.png', dpi=300, bbox_inches='tight')
