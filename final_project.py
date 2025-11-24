import os
import numpy as np
import kagglehub
import turtle
import random
import time
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

class SpermQuantumSimulation:
    def __init__(self, dataset_handle="orvile/mhsma-sperm-morphology-analysis-dataset"):
        self.dataset_handle = dataset_handle
        self.path = None
        self.prob_normal = 0.5
        self.prob_abnormal = 0.5
        self.counts = {}

    def fetch_data(self):
        print(f"--- Fetching Data ---")
        try:
            self.path = kagglehub.dataset_download(self.dataset_handle)
            print(f"Dataset path: {self.path}")
        except Exception as e:
            print(f"Error downloading: {e}")

    def calculate_bio_amplitudes(self):
        if not self.path:
            # Fallback if download fails or isn't run
            self.prob_normal = 0.52 
            self.prob_abnormal = 0.48
            return

        label_file = "y_head_train.npy"
        full_path = os.path.join(self.path, label_file)
        
        # Search for file
        if not os.path.exists(full_path):
            for root, dirs, files in os.walk(self.path):
                if label_file in files:
                    full_path = os.path.join(root, label_file)
                    break

        try:
            print(f"--- Loading Biological Data from {label_file} ---")
            labels = np.load(full_path)
            total = len(labels)
            abnormal = np.sum(labels)
            normal = total - abnormal

            self.prob_normal = normal / total
            self.prob_abnormal = abnormal / total
            print(f"Bio-probs -> Normal: {self.prob_normal:.2f}, Abnormal: {self.prob_abnormal:.2f}")

        except Exception as e:
            print(f"Data load error: {e}. Using default values.")

    def run_simulation(self, shots=1000):
        alpha = np.sqrt(self.prob_normal)
        theta = 2 * np.arccos(alpha)
        
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        qc.measure_all()

        simulator = AerSimulator()
        result = simulator.run(qc, shots=shots).result()
        self.counts = result.get_counts()
        return self.counts

class TurtleHeatmap:
    """
    A class to visualize Quantum Data using the Turtle library.
    It manually constructs a grid 'heatmap' based on probability density.
    """
    def __init__(self, grid_size=20, cell_size=20):
        self.grid_size = grid_size  # 20x20 grid = 400 cells
        self.cell_size = cell_size
        self.total_cells = grid_size * grid_size

    def draw_cell(self, t, color, x, y):
        """Helper to draw a single square"""
        t.penup()
        t.goto(x, y)
        t.pendown()
        t.fillcolor(color)
        t.begin_fill()
        for _ in range(4):
            t.forward(self.cell_size)
            t.right(90)
        t.end_fill()

    def visualize(self, counts):
        print("\n--- Starting Turtle Visualization ---")
        print("Check your taskbar if a window doesn't appear!")
        
        # 1. Process Data for the Grid
        # Calculate how many cells should be Blue (Normal) vs Red (Abnormal)
        total_shots = sum(counts.values())
        norm_count = counts.get('0', 0)
        
        # Scale to our grid size (e.g. 400 cells)
        num_blue = int((norm_count / total_shots) * self.total_cells)
        num_red = self.total_cells - num_blue
        
        # Create a list of colors representing the sperm pool
        # "cyan" for Normal (cool/safe), "red" for Abnormal (hot/warning)
        color_data = ["cyan"] * num_blue + ["red"] * num_red
        random.shuffle(color_data) # Shuffle to simulate random distribution in the pool

        # 2. Setup Turtle
        screen = turtle.Screen()
        screen.title(f"Quantum Sperm Heatmap (N={norm_count} vs Ab={total_shots-norm_count})")
        screen.bgcolor("black")
        screen.setup(width=600, height=600)
        # Turn off tracer for instant drawing (remove this line to see it draw one by one)
        screen.tracer(0) 

        t = turtle.Turtle()
        t.speed(0)
        t.hideturtle()
        
        # Start position (Top Left)
        start_x = -(self.grid_size * self.cell_size) / 2
        start_y = (self.grid_size * self.cell_size) / 2

        # 3. Draw the Grid
        idx = 0
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if idx < len(color_data):
                    x_pos = start_x + (col * self.cell_size)
                    y_pos = start_y - (row * self.cell_size)
                    
                    self.draw_cell(t, color_data[idx], x_pos, y_pos)
                    idx += 1
        
        # Update screen to show drawing
        screen.update()
        
        # Keep window open
        print("Visualization complete. Click the window to exit.")
        screen.exitonclick()

# --- Execution ---
if __name__ == "__main__":
    # 1. Run Quantum Simulation
    sim = SpermQuantumSimulation()
    sim.fetch_data()
    sim.calculate_bio_amplitudes()
    results = sim.run_simulation(shots=1000)
    
    # 2. Run Turtle Visualization
    # Note: We scale the heatmap to a 25x25 grid (625 samples) for better visibility
    viz = TurtleHeatmap(grid_size=25, cell_size=15)
    viz.visualize(results)