import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
from scipy.signal import savgol_filter
import re

# Constants for electric vehicle power model
VEHICLE_MASS = 1600  # kg (Tesla Model 3 approximate mass)
AIR_DENSITY = 1.2256  # kg/m^3
FRONTAL_AREA = 2.22  # m^2
DRAG_COEFFICIENT = 0.28
ROLLING_RESISTANCE = 0.011
GRAVITY = 9.81  # m/s^2

def calculate_jerk(acceleration, time_steps):
    """Calculate jerk (derivative of acceleration) using central difference."""
    jerk = np.zeros_like(acceleration)
    
    # Use central difference for interior points
    for i in range(1, len(acceleration) - 1):
        dt_prev = time_steps[i] - time_steps[i-1]
        dt_next = time_steps[i+1] - time_steps[i]
        jerk[i] = (acceleration[i+1] - acceleration[i-1]) / (dt_prev + dt_next)
    
    # Forward difference for first point
    if len(acceleration) > 1:
        jerk[0] = (acceleration[1] - acceleration[0]) / (time_steps[1] - time_steps[0])
    
    # Backward difference for last point
    if len(acceleration) > 1:
        jerk[-1] = (acceleration[-1] - acceleration[-2]) / (time_steps[-1] - time_steps[-2])
    
    return jerk

def calculate_steering_rate(steering, time_steps):
    """Calculate steering rate (derivative of steering angle)."""
    steering_rate = np.zeros_like(steering)
    
    for i in range(1, len(steering)):
        dt = time_steps[i] - time_steps[i-1]
        if dt > 0:
            steering_rate[i] = (steering[i] - steering[i-1]) / dt
    
    return steering_rate

def calculate_electric_power(mass, velocity, acceleration):
    """Calculate electric power consumption using the provided model."""
    # P_elect(t) = [ma(t) + (1/2)ρ_air·A_f·C_d·v^2(t) + mg·C_r]·v(t)
    aerodynamic_drag = 0.5 * AIR_DENSITY * FRONTAL_AREA * DRAG_COEFFICIENT * velocity**2
    rolling_resistance = mass * GRAVITY * ROLLING_RESISTANCE
    
    # Clamp negative acceleration (braking) since regenerative braking is not modeled here
    propulsion_force = mass * np.maximum(acceleration, 0)
    
    total_force = propulsion_force + aerodynamic_drag + rolling_resistance
    power = total_force * velocity  # Power = Force × Velocity
    
    return power

def calculate_energy_consumption(power, time_steps):
    """Calculate cumulative energy consumption in kWh."""
    energy = np.zeros_like(power)
    
    for i in range(1, len(power)):
        dt = time_steps[i] - time_steps[i-1]
        avg_power = (power[i] + power[i-1]) / 2  # Trapezoidal rule
        energy[i] = energy[i-1] + (avg_power * dt) / 3600000  # Convert W·s to kWh
    
    return energy

def calculate_metrics(df):
    """Calculate all metrics for a single dataset."""
    metrics = {}
    
    # Extract time and state variables
    time = df['time'].values
    ego_speed = df['ego_speed'].values
    ego_accel = df['ego_accel'].values
    ego_y = df['ego_y'].values
    control_steer = df['control_steer'].values
    
    # Calculate derivatives
    jerk = calculate_jerk(ego_accel, time)
    steering_rate = calculate_steering_rate(control_steer, time)
    
    # Target speed (from the MPC controller setting)
    target_speed = 25 / 3.6  # 25 km/h converted to m/s
    
    # Calculate speed tracking error
    speed_error = target_speed - ego_speed
    rms_speed_error = np.sqrt(np.mean(speed_error**2))
    
    # Calculate comfort metrics
    rms_jerk = np.sqrt(np.mean(jerk**2))
    rms_steering_rate = np.sqrt(np.mean(steering_rate**2))
    
    # Calculate lateral deviation (from reference trajectory at y=0)
    rms_lateral_deviation = np.sqrt(np.mean(ego_y**2))
    
    # Calculate power and energy consumption
    power = calculate_electric_power(VEHICLE_MASS, ego_speed, ego_accel)
    energy = calculate_energy_consumption(power, time)
    total_energy = energy[-1]  # Total energy consumed during maneuver
    
    # Overtaking time (time to travel 100m)
    overtaking_time = time[-1]
    
    # Store metrics
    metrics['rms_speed_error'] = rms_speed_error
    metrics['rms_jerk'] = rms_jerk
    metrics['rms_steering_rate'] = rms_steering_rate
    metrics['rms_lateral_deviation'] = rms_lateral_deviation
    metrics['total_energy'] = total_energy
    metrics['overtaking_time'] = overtaking_time
    
    # Store time series for plots
    metrics['time'] = time
    metrics['ego_speed'] = ego_speed
    metrics['ego_accel'] = ego_accel
    metrics['jerk'] = jerk
    metrics['ego_y'] = ego_y
    metrics['control_steer'] = control_steer
    metrics['steering_rate'] = steering_rate
    metrics['power'] = power
    metrics['energy'] = energy
    metrics['ego_x'] = df['ego_x'].values
    metrics['preceding_x'] = df['preceding_x'].values
    metrics['preceding_y'] = df['preceding_y'].values
    
    # ADD THESE LINES to capture target waypoint data
    if 'target_waypoint_x' in df.columns and 'target_waypoint_y' in df.columns:
        metrics['target_waypoint_x'] = df['target_waypoint_x'].values
        metrics['target_waypoint_y'] = df['target_waypoint_y'].values
    
    return metrics

def smooth_data(data, window_length=11, polyorder=3):
    """Apply Savitzky-Golay filter to smooth noisy data."""
    if len(data) < window_length:
        return data
    return savgol_filter(data, window_length, polyorder)

def extract_algorithm_name(filename):
    """Extract algorithm name from filename."""
    # Try to extract algorithm name using regex patterns
    patterns = [
        r'overtaking_simulation_\d+_(\w+)\.csv',  # Format: overtaking_simulation_20250423_MPC.csv
        r'(\w+)_overtaking_\d+\.csv',             # Format: MPC_overtaking_20250423.csv
        r'(\w+)_simulation\.csv'                  # Format: MPC_simulation.csv
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return match.group(1).upper()
    
    # If no pattern matches, use the filename without extension
    base = os.path.basename(filename)
    return os.path.splitext(base)[0]

def plot_trajectories(metrics_dict, output_dir):
    """Plot vehicle trajectories with reference path extrapolated correctly to longitude 100."""
    plt.figure(figsize=(10, 6))
    
    # Check if metrics dictionary is empty
    if not metrics_dict:
        print("No metrics data available for plotting trajectories.")
        return
    
    # Choose the first algorithm's data for fitting the reference path
    first_algo = list(metrics_dict.keys())[0]
    preceding_x = metrics_dict[first_algo]['preceding_x']
    preceding_y = metrics_dict[first_algo]['preceding_y']
    
    # Prepare data for fitting by sorting and removing duplicates
    sorted_indices = np.argsort(preceding_x)
    sorted_x = preceding_x[sorted_indices]
    sorted_y = preceding_y[sorted_indices]
    
    unique_indices = np.unique(sorted_x, return_index=True)[1]
    unique_x = sorted_x[unique_indices]
    unique_y = sorted_y[unique_indices]
    
    if len(unique_x) > 3:
        # Option 1: Use a quadratic polynomial (degree 2) instead of cubic
        # This typically results in a simpler curve with less unexpected behavior
        poly_coeffs = np.polyfit(unique_x, unique_y, 2)  # Changed from degree 3 to 2
        poly_fit = np.poly1d(poly_coeffs)
        
        # Option 2: Use linear extrapolation for the final segment
        # Get the direction of the last segment of data to ensure downward trend
        last_points_x = unique_x[-min(5, len(unique_x)):]
        last_points_y = unique_y[-min(5, len(unique_y)):]
        
        # Calculate linear trend for extrapolation
        end_slope, end_intercept = np.polyfit(last_points_x, last_points_y, 1)
        
        # Determine the transition point where we switch from polynomial to linear
        # Use the maximum x value from the data as the transition point
        x_max = np.max(preceding_x)
        
        # Generate extrapolated paths - polynomial for observed range, linear for extrapolation
        # First part: from minimum x to maximum observed x (use polynomial)
        x_min = np.min(preceding_x)
        observed_x = np.linspace(x_min, x_max, 100)
        observed_y = poly_fit(observed_x)
        
        # Second part: from maximum observed x to longitude 100 (linear with enforced trend)
        # If end_slope is not negative, force it to be slightly negative to ensure downward trend
        if end_slope >= 0:
            end_slope = -0.05  # Force a slight downward slope
        
        extrapolated_x = np.linspace(x_max, 100, 100)
        # y = mx + b, where m is the slope and b is adjusted to ensure continuity at x_max
        last_poly_y = poly_fit(x_max)  # Polynomial y-value at transition point
        extrapolated_y = end_slope * (extrapolated_x - x_max) + last_poly_y
        
        # Combine the two parts
        extrap_x = np.concatenate([observed_x, extrapolated_x])
        extrap_y = np.concatenate([observed_y, extrapolated_y])
        
        # Plot the improved reference path
        plt.plot(extrap_x, extrap_y, 'k--', linewidth=2, 
                label='Extrapolated Reference Path')
    
    # Plot actual trajectories for all algorithms
    for algo, metrics in metrics_dict.items():
        plt.plot(metrics['ego_x'], metrics['ego_y'], label=f'{algo} - Ego Vehicle')
        plt.scatter(metrics['preceding_x'], metrics['preceding_y'], 
                   marker='x', alpha=0.5, label=f'{algo} - Preceding Vehicle')
    
    plt.xlabel('Longitudinal Position (m)')
    plt.ylabel('Lateral Position (m)')
    plt.title('Vehicle Trajectories with Extended Reference Path')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'trajectories.png'), dpi=300, bbox_inches='tight')
    plt.close()

    
def plot_speed_profiles(metrics_dict, output_dir):
    """Plot speed profiles for all algorithms."""
    plt.figure(figsize=(10, 6))
    
    for algo, metrics in metrics_dict.items():
        plt.plot(metrics['time'], metrics['ego_speed'], label=f'{algo}')
    
    # Plot target speed
    target_speed = 25 / 3.6  # 25 km/h converted to m/s
    plt.axhline(y=target_speed, color='k', linestyle='--', label='Target Speed')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.title('Ego Vehicle Speed Profiles')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'speed_profiles.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_comfort_metrics(metrics_dict, output_dir):
    """Plot comfort metrics (jerk and steering rate) for all algorithms."""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot acceleration
    ax1 = fig.add_subplot(gs[0, 0])
    for algo, metrics in metrics_dict.items():
        ax1.plot(metrics['time'], metrics['ego_accel'], label=f'{algo}')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.set_title('Acceleration Profiles')
    ax1.grid(True)
    ax1.legend()
    
    # Plot jerk
    ax2 = fig.add_subplot(gs[0, 1])
    for algo, metrics in metrics_dict.items():
        # Smooth jerk data which can be noisy
        smoothed_jerk = smooth_data(metrics['jerk'])
        ax2.plot(metrics['time'], smoothed_jerk, label=f'{algo}')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Jerk (m/s³)')
    ax2.set_title('Jerk Profiles')
    ax2.grid(True)
    ax2.legend()
    
    # Plot steering angle
    ax3 = fig.add_subplot(gs[1, 0])
    for algo, metrics in metrics_dict.items():
        ax3.plot(metrics['time'], metrics['control_steer'], label=f'{algo}')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Steering Angle')
    ax3.set_title('Steering Profiles')
    ax3.grid(True)
    ax3.legend()
    
    # Plot steering rate
    ax4 = fig.add_subplot(gs[1, 1])
    for algo, metrics in metrics_dict.items():
        # Smooth steering rate data
        smoothed_steering_rate = smooth_data(metrics['steering_rate'])
        ax4.plot(metrics['time'], smoothed_steering_rate, label=f'{algo}')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Steering Rate (1/s)')
    ax4.set_title('Steering Rate Profiles')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comfort_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_lateral_deviation(metrics_dict, output_dir):
    """
    Plot lateral deviation showing target waypoints and actual path.
    
    This function visualizes:
    1. The target waypoints that the controller was following
    2. The lateral position of the ego vehicle(s) over time
    """
    # Create a figure with appropriate dimensions
    plt.figure(figsize=(10, 6))
    
    # Defensive programming: Check for empty data
    if not metrics_dict:
        print("No metrics data available for plotting lateral deviation.")
        return
    
    # Plot target waypoints for each algorithm
    for algo, metrics in metrics_dict.items():
        if 'target_waypoint_y' in metrics and 'time' in metrics:
            plt.scatter(metrics['time'], metrics['target_waypoint_y'], 
                       marker='o', s=30, alpha=0.5,
                       label=f'{algo} - Target Waypoints')
        else:
            print(f"Warning: Target waypoint data not found for {algo}")
        
    # Plot ego vehicle lateral positions for each control algorithm
    for algo, metrics in metrics_dict.items():
        plt.plot(metrics['time'], metrics['ego_y'], label=f'{algo} - Ego Path')
        
    # Finalize plot with proper formatting and annotations
    plt.xlabel('Time (s)')
    plt.ylabel('Lateral Position (m)')
    plt.title('Lateral Deviation with Target Waypoints')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'lateral_deviation.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_power_consumption(metrics_dict, output_dir):
    """Plot power and cumulative energy consumption for all algorithms."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot instantaneous power
    for algo, metrics in metrics_dict.items():
        # Smooth power data
        smoothed_power = smooth_data(metrics['power'])
        ax1.plot(metrics['time'], smoothed_power / 1000, label=f'{algo}')  # Convert to kW
    
    ax1.set_ylabel('Power (kW)')
    ax1.set_title('Instantaneous Power Consumption')
    ax1.grid(True)
    ax1.legend()
    
    # Plot cumulative energy
    for algo, metrics in metrics_dict.items():
        ax2.plot(metrics['time'], metrics['energy'], label=f'{algo}')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Energy (kWh)')
    ax2.set_title('Cumulative Energy Consumption')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'power_consumption.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_summary_metrics(metrics_dict, output_dir):
    """Create bar plots for summary metrics comparison."""
    # Extract scalar metrics for comparison
    algorithms = list(metrics_dict.keys())
    rms_speed_error = [metrics_dict[algo]['rms_speed_error'] for algo in algorithms]
    rms_jerk = [metrics_dict[algo]['rms_jerk'] for algo in algorithms]
    rms_steering_rate = [metrics_dict[algo]['rms_steering_rate'] for algo in algorithms]
    rms_lateral_deviation = [metrics_dict[algo]['rms_lateral_deviation'] for algo in algorithms]
    total_energy = [metrics_dict[algo]['total_energy'] for algo in algorithms]
    overtaking_time = [metrics_dict[algo]['overtaking_time'] for algo in algorithms]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # RMS Speed Error
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(algorithms, rms_speed_error)
    ax1.set_ylabel('RMS Error (m/s)')
    ax1.set_title('Speed Tracking Error')
    ax1.grid(axis='y')
    
    # RMS Jerk
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(algorithms, rms_jerk)
    ax2.set_ylabel('RMS Jerk (m/s³)')
    ax2.set_title('Passenger Comfort - Acceleration')
    ax2.grid(axis='y')
    
    # RMS Steering Rate
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(algorithms, rms_steering_rate)
    ax3.set_ylabel('RMS Steering Rate (1/s)')
    ax3.set_title('Passenger Comfort - Steering')
    ax3.grid(axis='y')
    
    # RMS Lateral Deviation
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(algorithms, rms_lateral_deviation)
    ax4.set_ylabel('RMS Deviation (m)')
    ax4.set_title('Path Following Accuracy')
    ax4.grid(axis='y')
    
    # Total Energy
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar(algorithms, total_energy)
    ax5.set_ylabel('Energy (kWh)')
    ax5.set_title('Total Energy Consumption')
    ax5.grid(axis='y')
    
    # Overtaking Time
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.bar(algorithms, overtaking_time)
    ax6.set_ylabel('Time (s)')
    ax6.set_title('Time to Complete Overtaking')
    ax6.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save metrics as a CSV file
    metrics_df = pd.DataFrame({
        'Algorithm': algorithms,
        'RMS_Speed_Error': rms_speed_error,
        'RMS_Jerk': rms_jerk,
        'RMS_Steering_Rate': rms_steering_rate,
        'RMS_Lateral_Deviation': rms_lateral_deviation,
        'Total_Energy_kWh': total_energy,
        'Overtaking_Time_s': overtaking_time
    })
    
    metrics_df.to_csv(os.path.join(output_dir, 'summary_metrics.csv'), index=False)
    print(f"Summary metrics saved to {os.path.join(output_dir, 'summary_metrics.csv')}")

def main():
    """Main function to process files and generate plots."""
    parser = argparse.ArgumentParser(description='Generate plots from simulation data files.')
    parser.add_argument('--input_dir', type=str, default='results',
                       help='Directory containing simulation CSV files')
    parser.add_argument('--output_dir', type=str, default='plots',
                       help='Directory to save generated plots')
    parser.add_argument('--files', type=str, nargs='+',
                       help='Specific CSV files to analyze (optional)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of files to analyze
    if args.files:
        file_list = args.files
    else:
        file_list = glob.glob(os.path.join(args.input_dir, '*.csv'))
    
    if not file_list:
        print(f"No CSV files found in {args.input_dir}")
        return
    
    print(f"Analyzing {len(file_list)} files:")
    for f in file_list:
        print(f"  - {f}")
    
    # Dictionary to store metrics for each algorithm
    metrics_dict = {}
    
    # Process each file
    for file_path in file_list:
        try:
            # Load data
            df = pd.read_csv(file_path)
            
            # Extract algorithm name from filename
            algo_name = extract_algorithm_name(os.path.basename(file_path))
            
            print(f"Processing {algo_name} data from {file_path}")
            
            # Calculate metrics
            metrics = calculate_metrics(df)
            
            # Store metrics
            metrics_dict[algo_name] = metrics
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if not metrics_dict:
        print("No valid data files were processed. Exiting.")
        return
    
    # Generate plots
    print("Generating plots...")
    
    # Vehicle trajectories
    plot_trajectories(metrics_dict, args.output_dir)
    
    # Speed profiles
    plot_speed_profiles(metrics_dict, args.output_dir)
    
    # Comfort metrics
    plot_comfort_metrics(metrics_dict, args.output_dir)

    # Lateral deviation
    if metrics_dict:
        print("Plotting lateral deviation...")
        plot_lateral_deviation(metrics_dict, args.output_dir)
    else:
        print("No valid data available for lateral deviation plot.")
    
    # Power consumption
    plot_power_consumption(metrics_dict, args.output_dir)
    
    # Summary metrics comparison
    plot_summary_metrics(metrics_dict, args.output_dir)
    
    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()