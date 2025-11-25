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
        self.selected_file = "Unknown"

    def fetch_data(self):
        print(f"--- Fetching Data ---")
        try:
            self.path = kagglehub.dataset_download(self.dataset_handle)
            print(f"Dataset downloaded to: {self.path}")
        except Exception as e:
            print(f"Error downloading: {e}")

    def calculate_bio_amplitudes(self):
        if not self.path:
            print("No path found. Using default values.")
            return

        # --- NEW: Random File Selection Logic ---
        # 1. Find ALL valid label files (starting with 'y_')
        label_files = []
        for root, dirs, files in os.walk(self.path):
            for file in files:
                # We only want label files (y_), not image files (x_)
                if file.startswith("y_") and file.endswith(".npy"):
                    label_files.append(os.path.join(root, file))
        
        if not label_files:
            print("No label files (y_*.npy) found!")
            return

        # 2. Pick one randomly
        selected_path = random.choice(label_files)
        self.selected_file = os.path.basename(selected_path)
        print(f"\n---> RANDOMLY SELECTED FILE: {self.selected_file}")

        # 3. Process the data
        try:
            labels = np.load(selected_path)
            total = len(labels)
            abnormal = np.sum(labels)
            normal = total - abnormal

            self.prob_normal = normal / total
            self.prob_abnormal = abnormal / total
            print(f"Stats -> Normal: {self.prob_normal:.2%}, Abnormal: {self.prob_abnormal:.2%}")

        except Exception as e:
            print(f"Error reading file: {e}")

    def run_simulation(self, shots=1000):
        # Prevent math errors if probability is 0
        if self.prob_normal <= 0: alpha = 0
        else: alpha = np.sqrt(self.prob_normal)
        
        theta = 2 * np.arccos(alpha)
        
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        qc.measure_all()

        simulator = AerSimulator()
        result = simulator.run(qc, shots=shots).result()
        self.counts = result.get_counts()
        return self.counts

class TurtleHeatmap:
    def __init__(self, grid_size=20, cell_size=20):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.total_cells = grid_size * grid_size

    def draw_square(self, t, color, x, y, size):
        """Helper to draw a filled square at x,y"""
        t.penup()
        t.goto(x, y)
        t.pendown()
        t.color("white", color) # White border, filled with 'color'
        t.begin_fill()
        for _ in range(4):
            t.forward(size)
            t.right(90)
        t.end_fill()

    def draw_legend(self, t, x, y):
        """Draws a professional legend on the side"""
        # Legend Box
        t.penup()
        t.goto(x, y)
        t.write("LEGEND", align="left", font=("Arial", 12, "bold"))
        
        # Item 1: Normal
        self.draw_square(t, "#00FFFF", x, y - 20, 15) # Cyan hex
        t.penup()
        t.goto(x + 25, y - 32)
        t.write("Normal (|0>)", align="left", font=("Arial", 10, "normal"))

        # Item 2: Abnormal
        self.draw_square(t, "#FF4444", x, y - 50, 15) # Soft Red hex
        t.penup()
        t.goto(x + 25, y - 62)
        t.write("Abnormal (|1>)", align="left", font=("Arial", 10, "normal"))

    def visualize(self, counts, filename):
        print("\n--- Generating Heatmap ---")
        screen = turtle.Screen()
        screen.setup(width=800, height=600)
        screen.bgcolor("#222222") # Dark mode background
        screen.title(f"Quantum Simulation - {filename}")
        screen.tracer(0) # Turn off animation for instant drawing

        t = turtle.Turtle()
        t.hideturtle()
        t.speed(0)
        t.pencolor("white")

        # 1. Prepare Data
        total_shots = sum(counts.values())
        norm_count = counts.get('0', 0)
        
        # Map quantum results to grid pixels
        num_blue = int((norm_count / total_shots) * self.total_cells)
        num_red = self.total_cells - num_blue
        
        # Color Palette (Hex codes are more professional)
        colors = ["#00FFFF"] * num_blue + ["#FF4444"] * num_red
        random.shuffle(colors)

        # 2. Draw Title
        t.penup()
        t.goto(0, 250)
        t.color("white")
        t.write(f"Quantum Distribution Map: {filename}", align="center", font=("Verdana", 16, "bold"))
        t.goto(0, 230)
        t.write(f"N={total_shots} | Normal: {norm_count} | Abnormal: {total_shots-norm_count}", 
                align="center", font=("Verdana", 10, "italic"))

        # 3. Draw Grid
        start_x = -200
        start_y = 200
        
        idx = 0
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if idx < len(colors):
                    x_pos = start_x + (col * self.cell_size)
                    y_pos = start_y - (row * self.cell_size)
                    self.draw_square(t, colors[idx], x_pos, y_pos, self.cell_size)
                    idx += 1

        # 4. Draw Legend (to the right of the grid)
        self.draw_legend(t, 250, 200)

        screen.update()
        print("Visualization Ready. Click window to exit.")
        screen.exitonclick()

if __name__ == "__main__":
    sim = SpermQuantumSimulation()
    sim.fetch_data()
    sim.calculate_bio_amplitudes()
    
    # Only run if we actually selected a file
    if sim.selected_file != "Unknown":
        results = sim.run_simulation(shots=1000)
        
        viz = TurtleHeatmap(grid_size=20, cell_size=20)
        viz.visualize(results, sim.selected_file)