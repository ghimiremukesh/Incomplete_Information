import matplotlib.pyplot as plt
import numpy as np

def smooth_data(data, window_size=3):
    # Applying a simple moving average for smoothing
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_losses(*file_data, smooth=True, window_size=3):
    min_iterations = float('inf')

    for file_path, label in file_data:
        with open(file_path, 'r') as file:
            losses = [float(line.strip()) for line in file]

        min_iterations = min(min_iterations, len(losses))

    for file_path, label in file_data:
        with open(file_path, 'r') as file:
            losses = [float(line.strip()) for line in file]

        if len(losses) > min_iterations:
            losses = losses[:min_iterations]

        if smooth:
            losses = smooth_data(losses, window_size)

        iterations = range(1, len(losses) + 1)

        plt.plot(iterations, losses, label=label)

    plt.title('Smoothed Losses Over Iterations' if smooth else 'Losses Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.show()

# Example usage with two files and their labels
file_data_1 = ('logs/soccer_uncons_effort_square/t_1/checkpoints_dir/train_losses_final.txt', 'Squared Distance')
file_data_2 = ('logs/soccer_uncons_effort/t_1/checkpoints_dir/train_losses_final.txt', 'Distance')

plot_losses(file_data_1, file_data_2, smooth=True, window_size=50)







