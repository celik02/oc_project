import numpy as np
from numpy import unwrap


def calculate_rmse(predictions, ground_truth):
    """Calculate Root Mean Square Error between predictions and ground truth"""
    return np.sqrt(np.mean((predictions - ground_truth) ** 2))


def calculate_ekf_metrics(data, start_index=10):
    """Calculate comprehensive metrics for EKF performance evaluation"""

    # Position RMSE (2D)
    pos_errors = data['ekf_position'][start_index:] - data['gt_position'][start_index:]
    pos_rmse_x = calculate_rmse(pos_errors[:, 0], np.zeros_like(pos_errors[:, 0]))
    pos_rmse_y = calculate_rmse(pos_errors[:, 1], np.zeros_like(pos_errors[:, 1]))
    pos_rmse_2d = np.sqrt(np.mean(np.sum(pos_errors**2, axis=1)))

    # Yaw RMSE (need to handle angle wrapping)
    # First unwrap both angles to remove discontinuities
    unwrapped_gt_yaw = unwrap(data['gt_yaw'][start_index:])
    unwrapped_ekf_yaw = unwrap(data['ekf_yaw'][start_index:])
    # Align the starting points if needed
    yaw_offset = (unwrapped_gt_yaw[0] - unwrapped_ekf_yaw[0])
    aligned_ekf_yaw = unwrapped_ekf_yaw + yaw_offset
    yaw_rmse = calculate_rmse(aligned_ekf_yaw, unwrapped_gt_yaw)

    # Velocity RMSE
    vel_rmse = calculate_rmse(data['velocity'][start_index:], data['gt_velocity'][start_index:])
    vel_rmse_kmh = vel_rmse * 3.6  # Convert to km/h

    # Print results
    print("\n--- EKF Performance Metrics ---")
    print(f"Position RMSE (X): {pos_rmse_x:.3f} m")
    print(f"Position RMSE (Y): {pos_rmse_y:.3f} m")
    print(f"Position RMSE (2D): {pos_rmse_2d:.3f} m")
    print(f"Yaw RMSE: {yaw_rmse:.4f} rad ({np.degrees(yaw_rmse):.2f} deg)")
    print(f"Velocity RMSE: {vel_rmse:.3f} m/s ({vel_rmse_kmh:.2f} km/h)")

    # Calculate 95th percentile errors
    pos_error_magnitudes = np.sqrt(np.sum(pos_errors**2, axis=1))
    pos_95th = np.percentile(pos_error_magnitudes, 95)
    yaw_95th = np.percentile(np.abs(aligned_ekf_yaw - unwrapped_gt_yaw), 95)
    vel_95th = np.percentile(np.abs(data['velocity'][start_index:] - data['gt_velocity'][start_index:]), 95)

    print("\n--- 95th Percentile Errors ---")
    print(f"Position (95%): {pos_95th:.3f} m")
    print(f"Yaw (95%): {yaw_95th:.4f} rad ({np.degrees(yaw_95th):.2f} deg)")
    print(f"Velocity (95%): {vel_95th:.3f} m/s ({vel_95th*3.6:.2f} km/h)")

    return {
        'pos_rmse_x': pos_rmse_x,
        'pos_rmse_y': pos_rmse_y,
        'pos_rmse_2d': pos_rmse_2d,
        'yaw_rmse': yaw_rmse,
        'vel_rmse': vel_rmse,
        'pos_95th': pos_95th,
        'yaw_95th': yaw_95th,
        'vel_95th': vel_95th
    }


if __name__ == '__main__':
    # Load the data
    data = np.load('ekf_data_plot.npz')

    # Calculate metrics
    metrics = calculate_ekf_metrics(data, start_index=10)

    # You could also add this to your plotting script:
    '''
    # Add this at the end of your plot_ekf_new.py
    # Calculate and display performance metrics
    metrics = calculate_ekf_metrics(data, start_index=start_index)

    # Add metrics as text on the figure
    metrics_text = (
        f"Position RMSE: {metrics['pos_rmse_2d']:.3f} m\n"
        f"Yaw RMSE: {np.degrees(metrics['yaw_rmse']):.2f}°\n"
        f"Velocity RMSE: {metrics['vel_rmse']*3.6:.2f} km/h"
    )

    fig.text(0.02, 0.02, metrics_text, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8))
    '''
