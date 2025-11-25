import sys
import os
import io
import random
import numpy as np
import kagglehub

# Scientific & Quantum Libraries
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Visualization Libraries
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt
import pyqtgraph.opengl as gl 
from pycirclize import Circos
import matplotlib.pyplot as plt

# --- 1. QUANTUM SIMULATION BACKEND ---
class SpermQuantumSimulation:
    def __init__(self, dataset_handle="orvile/mhsma-sperm-morphology-analysis-dataset"):
        self.dataset_handle = dataset_handle
        self.path = None
        self.prob_normal = 0.5 
        self.prob_abnormal = 0.5
        self.selected_file = "Unknown"

    def fetch_and_process(self):
        print("--- Initializing Biological Data ---")
        try:
            self.path = kagglehub.dataset_download(self.dataset_handle)
            
            # Find all label files
            label_files = []
            for root, _, files in os.walk(self.path):
                for file in files:
                    if file.startswith("y_") and file.endswith(".npy"):
                        label_files.append(os.path.join(root, file))
            
            if not label_files:
                raise FileNotFoundError("No 'y_*.npy' files found.")

            # Random selection
            selected_path = random.choice(label_files)
            self.selected_file = os.path.basename(selected_path)
            
            # Load and Force Binary (0 or 1)
            raw_labels = np.load(selected_path)
            labels = (raw_labels > 0).astype(int) 

            total = len(labels)
            abnormal = np.sum(labels)
            self.prob_normal = (total - abnormal) / total
            self.prob_abnormal = abnormal / total
            
            print(f"File: {self.selected_file}")
            print(f"Bio-Probs: Normal={self.prob_normal:.2%}, Abnormal={self.prob_abnormal:.2%}")

        except Exception as e:
            print(f"Data Error: {e}. Using default 50/50.")

    def run_simulation(self, shots=2000):
        # Prevent math domain errors
        p_norm = max(0.0, min(1.0, self.prob_normal))
        
        alpha = np.sqrt(p_norm)
        theta = 2 * np.arccos(alpha)
        
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        qc.measure_all()

        simulator = AerSimulator()
        compiled_qc = transpile(qc, simulator)
        result = simulator.run(compiled_qc, shots=shots).result()
        return result.get_counts()


# --- 2. ADVANCED VISUALIZATION FRONTEND ---
class ModernVisualizer(QMainWindow):
    def __init__(self, simulation_results, filename, probs):
        super().__init__()
        self.setWindowTitle(f"Quantum Sperm Genomics - {filename}")
        self.resize(1200, 700)
        self.setStyleSheet("background-color: #1e1e1e; color: white;")

        # Data Parsing
        self.total_shots = sum(simulation_results.values())
        self.count_normal = (
            simulation_results.get('0', 0) + 
            simulation_results.get('0x0', 0) + 
            simulation_results.get(0, 0)
        )
        self.count_abnormal = self.total_shots - self.count_normal
        self.probs = probs 
        self.filename = filename

        # --- Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout() 
        central_widget.setLayout(layout)

        # --- LEFT PANEL: 3D View + Stats ---
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        
        # 1. OpenGL View
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setBackgroundColor('#101010')
        self.gl_view.setCameraPosition(distance=45, elevation=30)
        
        g = gl.GLGridItem()
        g.setSize(x=40, y=40)
        g.setSpacing(x=2, y=2)
        self.gl_view.addItem(g)
        
        self.create_gl_cloud()
        left_layout.addWidget(self.gl_view, stretch=3)

        # 2. UPDATED Stats Label
        # Calculations
        sim_n_pct = (self.count_normal / self.total_shots) * 100
        sim_a_pct = (self.count_abnormal / self.total_shots) * 100
        bio_n_pct = self.probs[0] * 100
        bio_a_pct = self.probs[1] * 100
        
        # Deviation (Quantum Noise)
        diff_n = sim_n_pct - bio_n_pct

        stats_text = (
            f"<h3 style='color: white;'>QUANTUM ANALYSIS REPORT</h3>"
            f"<b>Source Dataset:</b> {self.filename}<br>"
            f"<b>Total Samples:</b> {self.total_shots}<br>"
            f"--------------------------------------------------<br>"
            
            f"<b style='color: #00FFFF; font-size: 14px;'>■ NORMAL (Cyan)</b><br>"
            f"&nbsp;&nbsp;Measured Count: <b>{self.count_normal}</b><br>"
            f"&nbsp;&nbsp;Simulated Ratio: {sim_n_pct:.2f}%<br>"
            f"&nbsp;&nbsp;Biological Input: {bio_n_pct:.2f}%<br>"
            f"&nbsp;&nbsp;<i>Quantum Deviation: {diff_n:+.2f}%</i><br><br>"
            
            f"<b style='color: #FF4444; font-size: 14px;'>■ ABNORMAL (Red)</b><br>"
            f"&nbsp;&nbsp;Measured Count: <b>{self.count_abnormal}</b><br>"
            f"&nbsp;&nbsp;Simulated Ratio: {sim_a_pct:.2f}%<br>"
            f"&nbsp;&nbsp;Biological Input: {bio_a_pct:.2f}%"
        )

        stats_label = QLabel(stats_text)
        stats_label.setFont(QFont("Segoe UI", 10))
        stats_label.setStyleSheet("background-color: #252526; padding: 15px; border-top: 2px solid #333;")
        stats_label.setTextFormat(Qt.TextFormat.RichText)
        
        left_layout.addWidget(stats_label, stretch=1)
        layout.addWidget(left_panel, stretch=2)

        # --- RIGHT PANEL: Circular Plot ---
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.create_circlize_plot()
        layout.addWidget(self.image_label, stretch=1)

    def create_gl_cloud(self):
        """Renders the 3D scatter plot."""
        pos = np.random.normal(size=(self.total_shots, 3))
        pos[:, 0] *= 5  
        pos[:, 1] *= 5  
        pos[:, 2] *= 0.5 
        
        colors = np.zeros((self.total_shots, 4))
        
        if self.count_normal > 0:
            colors[:self.count_normal] = (0, 1, 1, 0.8) # Cyan
        if self.count_abnormal > 0:
            colors[self.count_normal:] = (1, 0.3, 0.3, 0.8) # Red

        np.random.shuffle(pos)

        sp = gl.GLScatterPlotItem(pos=pos, color=colors, size=6, pxMode=True)
        sp.setGLOptions('translucent')
        self.gl_view.addItem(sp)

    def create_circlize_plot(self):
        """Generates Circular Diagram."""
        n_pct = (self.count_normal / self.total_shots) * 100
        a_pct = (self.count_abnormal / self.total_shots) * 100

        sectors = {f"Normal\n{n_pct:.1f}%": n_pct, f"Abnormal\n{a_pct:.1f}%": a_pct}
        circos = Circos(sectors, space=5)

        for sector in circos.sectors:
            color = "#00FFFF" if "Normal" in sector.name else "#FF4444"
            track1 = sector.add_track((90, 100))
            track1.axis(fc=color, alpha=0.6)
            track1.text(sector.name, color="white", size=10)

            track2 = sector.add_track((60, 85))
            x = np.linspace(sector.start, sector.end, 40)
            y = np.random.randint(20, 100, 40)
            track2.bar(x, y, color=color, alpha=0.8)

        fig = circos.plotfig()
        fig.patch.set_facecolor('#1e1e1e')
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor='#1e1e1e', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        qimg = QImage.fromData(buf.getvalue())
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap)

# --- 3. EXECUTION ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    sim = SpermQuantumSimulation()
    sim.fetch_and_process()
    counts = sim.run_simulation(shots=2000)
    window = ModernVisualizer(counts, sim.selected_file, (sim.prob_normal, sim.prob_abnormal))
    window.show()
    sys.exit(app.exec())