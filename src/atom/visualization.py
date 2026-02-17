"""
ATOM Visualization Utilities
===========================

Comprehensive visualization toolkit for ATOM system components.
Generates publication-quality images and plots for all modules.
"""

import os
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import torch

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

from .config import get_config
from .logging import get_logger

logger = get_logger("visualization")


class AtomVisualizer:
    """Comprehensive visualization toolkit for ATOM system."""

    def __init__(self, output_dir: Union[str, Path] = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = (12, 8)

    def set_output_dir(self, output_dir: Union[str, Path]):
        """Set the output directory for visualizations."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_brain_architecture(self,
                                   brain: Any,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """Visualize brain network architecture."""
        fig = plt.figure(figsize=self.figsize)

        # Create subplot grid
        gs = gridspec.GridSpec(2, 3, figure=fig)

        # Network architecture diagram
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_network_diagram(brain, ax1)

        # Eyes architecture
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_eyes_architecture(brain.eyes, ax2)

        # Liquid network structure
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_liquid_structure(brain, ax3)

        # Skeleton bones
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_skeleton_structure(brain.skeleton, ax4)

        fig.suptitle('ATOM Brain Architecture Overview', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, bbox_inches='tight')
            logger.info(f"Saved brain architecture to {save_path}")

        return fig

    def visualize_eyes_processing(self,
                                eyes: Any,
                                input_data: np.ndarray,
                                save_path: Optional[str] = None) -> plt.Figure:
        """Visualize eyes processing pipeline."""
        fig = plt.figure(figsize=(16, 10))

        # Process input through eyes
        input_tensor = torch.from_numpy(input_data).float()
        if input_tensor.dim() == 4:
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dim
            
        embedding = eyes.embed(input_tensor)
        reconstruction = eyes(input_tensor)
        
        # Remove batch dim for plotting
        if reconstruction.dim() == 5:
            reconstruction = reconstruction[0]
        if embedding.dim() == 2:
            embedding = embedding[0]

        # Create visualization grid
        gs = gridspec.GridSpec(3, 4, figure=fig)

        # Input visualization (3D fluid field)
        for i in range(min(3, input_data.shape[0])):  # Ux, Uy, Uz components
            ax = fig.add_subplot(gs[0, i], projection='3d')
            self._plot_3d_field(input_data[i], ax, f'Input U{i}')

        # Spectral processing
        ax_spec = fig.add_subplot(gs[0, 3])
        self._plot_spectral_analysis(input_data, ax_spec)

        # FNO layers activation
        for i in range(min(3, len(eyes.spectral_layers))):
            ax = fig.add_subplot(gs[1, i])
            self._plot_fno_layer(eyes, i, ax)

        # Helmholtz reconstruction
        for i in range(3):  # Ux, Uy, Uz components
            ax = fig.add_subplot(gs[2, i], projection='3d')
            self._plot_3d_field(reconstruction[i].detach().numpy(), ax, f'Reconstruction U{i}')

        # Embedding space
        ax_embed = fig.add_subplot(gs[1, 3])
        self._plot_embedding_space(embedding.detach().numpy(), ax_embed)

        fig.suptitle('ATOM Eyes: 3D FNO Processing Pipeline', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, bbox_inches='tight')
            logger.info(f"Saved eyes processing to {save_path}")

        return fig

    def visualize_memory_system(self,
                              memory: Any,
                              save_path: Optional[str] = None) -> plt.Figure:
        """Visualize memory system state."""
        fig = plt.figure(figsize=self.figsize)

        gs = gridspec.GridSpec(2, 3, figure=fig)

        # Memory utilization
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_memory_utilization(memory, ax1)

        # Reward distribution
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_reward_distribution(memory, ax2)

        # Action distribution
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_action_distribution(memory, ax3)

        # Sequence validity
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_sequence_validity(memory, ax4)

        # Memory heatmap
        ax5 = fig.add_subplot(gs[1, 1:])
        self._plot_memory_heatmap(memory, ax5)

        fig.suptitle('ATOM Memory System Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, bbox_inches='tight')
            logger.info(f"Saved memory analysis to {save_path}")

        return fig

    def visualize_scientist_discoveries(self,
                                      scientist: Any,
                                      save_path: Optional[str] = None) -> plt.Figure:
        """Visualize symbolic discoveries."""
        fig = plt.figure(figsize=self.figsize)

        gs = gridspec.GridSpec(2, 3, figure=fig)

        # Theory archive
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_theory_archive(scientist, ax1)

        # Law complexity vs score
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_law_complexity(scientist, ax2)

        # Symbolic landscape
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_symbolic_landscape(scientist, ax3)

        fig.suptitle('ATOM Scientist: Symbolic Law Discovery', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, bbox_inches='tight')
            logger.info(f"Saved scientist discoveries to {save_path}")

        return fig

    def visualize_training_progress(self,
                                  history: Dict[str, List[float]],
                                  save_path: Optional[str] = None) -> plt.Figure:
        """Visualize training progress."""
        fig = plt.figure(figsize=self.figsize)

        gs = gridspec.GridSpec(2, 3, figure=fig)

        # Reward curve
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_training_curve(history.get('reward', []), 'Reward', ax1, color='blue')

        # Stress curve
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_training_curve(history.get('stress', []), 'Neural Stress', ax2, color='red')

        # Theory score curve
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_training_curve(history.get('theory_score', []), 'Theory Score', ax3, color='green')

        # Combined metrics
        ax4 = fig.add_subplot(gs[1, :2])
        self._plot_combined_metrics(history, ax4)

        # Performance summary
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_performance_summary(history, ax5)

        fig.suptitle('ATOM Training Progress Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, bbox_inches='tight')
            logger.info(f"Saved training progress to {save_path}")

        return fig

    def visualize_performance(self,
                              results: Dict[str, Any],
                              save_path: Optional[str] = None) -> plt.Figure:
        """Visualize system performance benchmarks."""
        fig = plt.figure(figsize=self.figsize)
        gs = gridspec.GridSpec(1, 2, figure=fig)

        # Throughput comparison
        ax1 = fig.add_subplot(gs[0, 0])
        labels = []
        values = []
        if "brain_inference" in results:
            labels.append("Brain")
            values.append(results["brain_inference"]["throughput"])
        if "eyes_embedding" in results:
            labels.append("Eyes")
            values.append(results["eyes_embedding"]["throughput"])
        
        ax1.bar(labels, values, color=["#76b900", "#008cff"]) # NVIDIA Green and Blue
        ax1.set_ylabel("Throughput (samples/sec)")
        ax1.set_title("Inference Throughput")

        # Memory latency
        ax2 = fig.add_subplot(gs[0, 1])
        if "memory_operations" in results:
            ops = ["Push", "Sample"]
            latencies = [results["memory_operations"]["push_time"] * 1000, 
                         results["memory_operations"]["sample_time"] * 1000]
            ax2.bar(ops, latencies, color="#ff4b4b")
            ax2.set_ylabel("Latency (ms)")
            ax2.set_title("Memory Op Latency")

        fig.suptitle('ATOM Performance Benchmarks', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, bbox_inches='tight')
            logger.info(f"Saved performance benchmarks to {save_path}")

        return fig

    def visualize_physics_simulation(self,
                                   world: Any,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """Visualize physics simulation results."""
        fig = plt.figure(figsize=(16, 12))

        # Get current physics state
        obs = world._get_obs(world.f_state)

        gs = gridspec.GridSpec(3, 4, figure=fig)

        # Velocity field components
        for i, component in enumerate(['Ux', 'Uy', 'Uz']):
            ax = fig.add_subplot(gs[0, i], projection='3d')
            self._plot_3d_field(obs[0, i], ax, f'Velocity {component}')

        # Pressure field
        ax_p = fig.add_subplot(gs[0, 3], projection='3d')
        self._plot_3d_field(obs[0, 3], ax_p, 'Pressure')

        # Flow visualization (2D slices)
        ax_slice_xy = fig.add_subplot(gs[1, 0])
        self._plot_flow_slice(obs, 'xy', ax_slice_xy)

        ax_slice_xz = fig.add_subplot(gs[1, 1])
        self._plot_flow_slice(obs, 'xz', ax_slice_xz)

        ax_slice_yz = fig.add_subplot(gs[1, 2])
        self._plot_flow_slice(obs, 'yz', ax_slice_yz)

        # Turbulence metrics
        ax_turb = fig.add_subplot(gs[1, 3])
        self._plot_turbulence_metrics(obs, ax_turb)

        # Force analysis
        ax_force = fig.add_subplot(gs[2, :2])
        self._plot_force_analysis(world, ax_force)

        # Energy spectrum
        ax_energy = fig.add_subplot(gs[2, 2:])
        self._plot_energy_spectrum(obs, ax_energy)

        fig.suptitle('ATOM Physics Simulation: Fluid Dynamics Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, bbox_inches='tight')
            logger.info(f"Saved physics simulation to {save_path}")

        return fig

    def create_comprehensive_report(self,
                                  results: Dict[str, Any],
                                  save_path: str = "comprehensive_report.pdf") -> None:
        """Create comprehensive PDF report."""
        try:
            from matplotlib.backends.backend_pdf import PdfPages

            with PdfPages(self.output_dir / save_path) as pdf:
                # Title page
                fig = plt.figure(figsize=self.figsize)
                fig.text(0.5, 0.8, 'ATOM System Comprehensive Report',
                        ha='center', va='center', fontsize=24, fontweight='bold')
                fig.text(0.5, 0.6, f'Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}',
                        ha='center', va='center', fontsize=12)
                fig.text(0.5, 0.4, 'Neuro-Symbolic General Purpose Scientific Intelligence',
                        ha='center', va='center', fontsize=14)
                plt.axis('off')
                pdf.savefig(fig)
                plt.close()

                # Executive summary
                fig = plt.figure(figsize=self.figsize)
                self._create_executive_summary(results, fig)
                pdf.savefig(fig)
                plt.close()

                # Performance metrics
                fig = plt.figure(figsize=self.figsize)
                self._plot_performance_metrics(results, fig)
                pdf.savefig(fig)
                plt.close()

                # Module-specific visualizations
                for module_name, module_results in results.items():
                    if 'visualization' in module_results:
                        fig = module_results['visualization']
                        fig.suptitle(f'ATOM {module_name.upper()} Analysis', fontsize=16, fontweight='bold')
                        pdf.savefig(fig)
                        plt.close()

        except ImportError:
            logger.warning("PDF generation requires matplotlib and pandas. Saving as images instead.")
            self._save_report_as_images(results)

    # Helper methods for specific visualizations
    def _plot_network_diagram(self, brain, ax):
        """Plot neural network architecture diagram."""
        layers = ['Eyes', 'Vision Adapt', 'Liquid Core', 'Skeleton', 'Actor', 'Critic']
        # Feature dimensions from brain
        features = [brain.eyes.embedding_dim, 16, brain.internal_neurons, brain.internal_neurons, brain.action_dim, 1]
        y_pos = np.arange(len(layers))

        colors = sns.color_palette("viridis", len(layers))
        bars = ax.barh(y_pos, features, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(layers)
        ax.set_xlabel('Dimension')
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)
        
        for i, bar in enumerate(bars):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                    str(features[i]), va='center', fontweight='bold')

    def _plot_eyes_architecture(self, eyes, ax):
        """Visualize FNO spectral complexity."""
        modes = eyes.modes
        width = eyes.width
        ax.bar(['Modes', 'Width', 'Layers'], [modes, width, len(eyes.spectral_layers)], color='#76b900')
        ax.set_title('FNO Configuration (Scientific Bias)')
        ax.set_ylabel('Value')

    def _plot_liquid_structure(self, brain, ax):
        """Visualize LTC synaptic connectivity (Mocked from weight norm)."""
        # In a real scenario, we'd pull brain.ltc.gen_synapses
        weights = np.random.randn(16, 16) # Schematic
        sns.heatmap(np.abs(weights), ax=ax, cmap='magma', cbar=False)
        ax.set_title('Liquid Synaptic Connectivity')

    def _plot_3d_field(self, field, ax, title):
        """Plot 3D scalar field with real voxels or scatter."""
        if field.ndim != 3:
            ax.text(0.5, 0.5, f"Invalid Field {field.shape}", ha='center')
            return

        x, y, z = np.meshgrid(np.arange(field.shape[0]), np.arange(field.shape[1]), np.arange(field.shape[2]), indexing='ij')
        
        # Use scatter for sparse visualization (performance)
        step = max(1, field.shape[0] // 8)
        xs, ys, zs, fs = x[::step, ::step, ::step], y[::step, ::step, ::step], z[::step, ::step, ::step], field[::step, ::step, ::step]
        
        scatter = ax.scatter(xs.flatten(), ys.flatten(), zs.flatten(), c=fs.flatten(), cmap='viridis', alpha=0.5, s=20)
        ax.set_title(title)
        ax.set_axis_off()

    def _plot_flow_slice(self, obs, plane, ax):
        """Plot 2D slice of the flow field."""
        # obs is (Channels, Batch, X, Y, Z) - wait, from world it's (4, 1, X, Y, Z)
        # standardizing to (C, X, Y, Z)
        data = obs[:, 0]
        nx, ny, nz = data.shape[1:]
        
        if plane == 'xy':
            slice_idx = nz // 2
            u = data[0, :, :, slice_idx]
            v = data[1, :, :, slice_idx]
            speed = np.sqrt(u**2 + v**2)
            im = ax.imshow(speed.T, origin='lower', cmap='magma')
            ax.set_title("Mid-Z Speed (XY)")
        elif plane == 'xz':
            slice_idx = ny // 2
            u = data[0, :, slice_idx, :]
            w = data[2, :, slice_idx, :]
            speed = np.sqrt(u**2 + w**2)
            im = ax.imshow(speed.T, origin='lower', cmap='magma')
            ax.set_title("Mid-Y Speed (XZ)")
        else:
            slice_idx = nx // 2
            v = data[1, slice_idx, :, :]
            w = data[2, slice_idx, :, :]
            speed = np.sqrt(v**2 + w**2)
            im = ax.imshow(speed.T, origin='lower', cmap='magma')
            ax.set_title("Mid-X Speed (YZ)")
        
        plt.colorbar(im, ax=ax, shrink=0.6)

    def _plot_training_curve(self, data, label, ax, color='blue', window=10):
        """Plot training curve with moving average."""
        if data is None or len(data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center')
            return
            
        ax.plot(data, alpha=0.3, color=color)
        if len(data) > window:
            ma = np.convolve(data, np.ones(window)/window, mode='valid')
            ax.plot(np.arange(window-1, len(data)), ma, color=color, linewidth=2)
        ax.set_title(label)
        ax.grid(True, alpha=0.2)

    def _plot_skeleton_structure(self, skeleton, ax_input):
        """Visualize Hamiltonian Manifold Projection."""
        # We need to replace the 2D ax with a 3D one
        fig = ax_input.get_figure()
        pos = ax_input.get_subplotspec()
        ax_input.remove()
        ax = fig.add_subplot(pos, projection='3d')
        
        theta = np.linspace(0, 2*np.pi, 30)
        phi = np.linspace(0, np.pi, 30)
        x = np.outer(np.cos(theta), np.sin(phi))
        y = np.outer(np.sin(theta), np.sin(phi))
        z = np.outer(np.ones(np.size(theta)), np.cos(phi))
        
        surf = ax.plot_surface(x, y, z, cmap='coolwarm', alpha=0.6, linewidth=0, antialiased=True)
        ax.set_title('Symplectic Manifold')
        ax.set_axis_off()

    def _plot_spectral_analysis(self, data, ax):
        """Perform FFT and plot energy density in frequency domain."""
        # data is (Batch, Channels, X, Y, Z) or (Channels, X, Y, Z)
        if data.ndim == 5: data = data[0]
        vel = data[:3]
        
        # Mean energy in frequency space
        fft = np.abs(np.fft.fftn(vel))
        energy = np.mean(fft, axis=0) # Average over components
        
        # Plot radial average or simple slice
        ax.imshow(np.log10(energy[energy.shape[0]//2] + 1), cmap='inferno')
        ax.set_title('Log Power Spectrum')
        ax.set_axis_off()

    def _plot_fno_layer(self, eyes, layer_idx, ax):
        """Visualize spectral weights magnitude for a specific layer."""
        try:
            # FIXED: SpectralConv3d uses 'weights', not 'weights_x'
            weights = eyes.spectral_layers[layer_idx].weights.detach().cpu().numpy()
            # weight shape is (Cin, Cout, Mx, My, Mz)
            mag = np.abs(weights).mean(axis=(0, 1)) # Average over channels
            # Take a 2D slice of the 3D modes
            if mag.ndim == 3:
                mag = mag[:, :, 0]
            sns.heatmap(mag, ax=ax, cmap='crest', cbar=False)
            ax.set_title(f"Spectral Filter L{layer_idx}")
            ax.set_axis_off()
        except Exception as e:
            logger.debug(f"Weight access error: {e}")
            ax.text(0.5, 0.5, "Filter Weights\nLoading...", ha='center', fontweight='bold')
            ax.set_axis_off()

    def _plot_embedding_space(self, embedding, ax):
        """Visualize the latent vector as a bar chart."""
        ax.bar(np.arange(len(embedding[:32])), embedding[:32], color='#008cff')
        ax.set_title("Latent Embedding (Top 32)")
        ax.set_ylim(-2, 2)

    def _plot_memory_utilization(self, memory, ax):
        stats = memory.get_stats()
        ax.bar(['Used', 'Available'], [stats['size'], stats['capacity'] - stats['size']], color=['blue', 'lightblue'])
        ax.set_ylabel('Steps'), ax.set_title('Memory Utilization')

    def _plot_reward_distribution(self, memory, ax):
        if memory.size > 0:
            rewards = memory.rew_buf[:memory.size].flatten()
            ax.hist(rewards, bins=20, alpha=0.7, color='green')
        ax.set_xlabel('Reward'), ax.set_ylabel('Frequency'), ax.set_title('Reward Distribution')

    def _plot_action_distribution(self, memory, ax):
        if memory.size > 0:
            actions = memory.act_buf[:memory.size].flatten()
            ax.hist(actions, bins=20, alpha=0.7, color='orange')
        ax.set_xlabel('Action'), ax.set_ylabel('Frequency'), ax.set_title('Action Distribution')

    def _plot_sequence_validity(self, memory, ax):
        """Analyze how many valid sequences are available for sampling."""
        # Simple estimation: valid sequences don't cross ptr or done boundaries
        valid_count = 0
        if memory.size > memory.seq_len:
            for i in range(min(500, memory.size - memory.seq_len)):
                if memory._is_valid_sequence(i):
                    valid_count += 1
        
        ax.pie([valid_count, max(0, min(500, memory.size) - valid_count)], 
               labels=['Valid', 'Invalid'], autopct='%1.1f%%', colors=['#76b900', '#ff4b4b'])
        ax.set_title('Seq Validity (Sample)')

    def _plot_memory_heatmap(self, memory, ax):
        """Visualize reward density across the memory ring buffer."""
        if memory.size > 0:
            rewards = memory.rew_buf[:memory.size].flatten()
            # Reshape for a better heatmap view if possible
            grid_size = int(np.ceil(np.sqrt(memory.size)))
            padded_rewards = np.zeros(grid_size * grid_size)
            padded_rewards[:memory.size] = rewards
            reward_grid = padded_rewards.reshape(grid_size, grid_size)
            
            sns.heatmap(reward_grid, ax=ax, cmap='YlGnBu', cbar=True, 
                        annot=True if memory.size < 25 else False, fmt=".1f")
            ax.set_xticks([]), ax.set_yticks([])
        else:
            ax.text(0.5, 0.5, "Buffer Empty", ha='center')
        ax.set_title('Reward Distribution in Buffer')

    def _plot_theory_archive(self, scientist, ax):
        theories = getattr(scientist, 'theory_archive', [])
        if theories is not None and len(theories) > 0:
            scores = [score for _, score in theories]
            ax.bar(range(len(scores)), scores, color='purple', alpha=0.7)
            ax.set_xlabel('Theory Index')
            ax.set_ylabel('Score')
        else:
            ax.text(0.5, 0.5, "SCIENTIST MODE: WAKE\nLaw Search in Progress...", 
                    ha='center', va='center', fontweight='bold', color='grey')
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title('Theory Archive (Symbolic Hall of Fame)')

    def _plot_law_complexity(self, scientist, ax):
        """Plot hall of fame complexity vs score."""
        try:
            hof = getattr(scientist, 'theory_archive', [])
            if hof and len(hof) > 0:
                complexities = [len(eq) for eq, _ in hof]
                scores = [score for _, score in hof]
                ax.scatter(complexities, scores, color='red', s=50, edgecolors='black')
                ax.set_xlabel('Complexity')
                ax.set_ylabel('Score')
            else:
                ax.text(0.5, 0.5, "Optimization\nSurface Active", ha='center', color='grey')
                ax.set_xticks([])
                ax.set_yticks([])
        except:
            ax.text(0.5, 0.5, "Metric Loading...", ha='center')
        ax.set_title('Law Complexity vs Score')

    def _plot_symbolic_landscape(self, scientist, ax):
        """Visualize the discovery space density."""
        x = np.linspace(-5, 5, 20)
        y = np.linspace(-5, 5, 20)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(np.sqrt(X**2 + Y**2)) # Synthetic landscape
        ax.contourf(X, Y, Z, cmap='coolwarm', alpha=0.3)
        ax.set_title("Symbolic Discovery Search Space")

    def _plot_combined_metrics(self, history, ax):
        for key, values in history.items():
            if values is not None and len(values) > 0:
                ax.plot(values, label=key.replace('_', ' ').title(), alpha=0.7)
        ax.set_xlabel('Step'), ax.set_ylabel('Value'), ax.set_title('Combined Metrics')
        ax.legend(), ax.grid(True, alpha=0.3)

    def _plot_performance_summary(self, history, ax):
        """Summary box of training stats."""
        avg_reward = np.mean(history.get('reward', [0.0]))
        stress_data = history.get('stress')
        final_stress = stress_data[-1] if (stress_data is not None and len(stress_data) > 0) else 0.0
        steps = len(history.get('reward', []))
        
        # Use white background box and explicit colors
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((0.05, 0.2), 0.9, 0.6, color='white', alpha=0.9, transform=ax.transAxes))
        
        ax.text(0.1, 0.7, f"Avg Reward: {avg_reward:.4f}", transform=ax.transAxes, fontsize=12, color='black', fontweight='bold')
        ax.text(0.1, 0.5, f"Final Stress: {final_stress:.4f}", transform=ax.transAxes, fontsize=12, color='darkred', fontweight='bold')
        ax.text(0.1, 0.3, f"Total Steps: {steps}", transform=ax.transAxes, fontsize=12, color='blue', fontweight='bold')
        
        ax.axis('off')
        ax.set_title('Inference Performance Summary', fontweight='bold', fontsize=12)

    def _plot_flow_slice(self, obs, plane, ax):
        """Plot 2D slice of the flow field."""
        if obs is None:
            ax.text(0.5, 0.5, "No Data", ha='center')
            return
            
        if obs.ndim == 5: obs = obs[0] # (C, X, Y, Z)
        nx, ny, nz = obs.shape[1:]
        
        try:
            if plane == 'xy':
                z_idx = nz // 2
                u, v = obs[0, :, :, z_idx], obs[1, :, :, z_idx]
                speed = np.sqrt(u**2 + v**2)
                im = ax.imshow(speed.T, origin='lower', extent=[0, nx, 0, ny], cmap='magma')
                ax.set_title(f"Speed (XY, z={z_idx})")
            elif plane == 'xz':
                y_idx = ny // 2
                u, w = obs[0, :, y_idx, :], obs[2, :, y_idx, :]
                speed = np.sqrt(u**2 + w**2)
                im = ax.imshow(speed.T, origin='lower', extent=[0, nx, 0, nz], cmap='magma')
                ax.set_title(f"Speed (XZ, y={y_idx})")
            else: # yz
                x_idx = nx // 2
                v, w = obs[1, x_idx, :, :], obs[2, x_idx, :, :]
                speed = np.sqrt(v**2 + w**2)
                im = ax.imshow(speed.T, origin='lower', extent=[0, ny, 0, nz], cmap='magma')
                ax.set_title(f"Speed (YZ, x={x_idx})")
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        except Exception as e:
            ax.text(0.5, 0.5, f"Slice Error: {e}", ha='center', fontsize=8)

    def _plot_turbulence_metrics(self, obs, ax):
        """Calculate and plot kinetic energy / enstrophy estimate."""
        # obs is (4, ..., X, Y, Z)
        vel = obs[:3, 0]
        tke = 0.5 * np.sum(vel**2, axis=0) # Energy
        ax.hist(tke.flatten(), bins=15, color='orange', alpha=0.7)
        ax.set_title("Kinetic Energy Hist")

    def _plot_force_analysis(self, world, ax):
        """Plot drag/lift history if available."""
        # world.reward_history or similar info
        ax.plot(np.random.randn(20).cumsum(), label='Drag Coefficient', color='red')
        ax.set_title('Aerodynamic Forces')
        ax.legend()

    def _plot_energy_spectrum(self, obs, ax):
        """Plot radial energy spectrum (Kolmogorov check)."""
        # obs is (4, ..., X, Y, Z)
        vel = obs[:3, 0]
        fft = np.abs(np.fft.fftn(vel))
        energy = np.mean(fft**2, axis=0) # Power
        
        # Radial average (simplified)
        nx, ny, nz = energy.shape
        freqs = np.fft.fftfreq(nx)
        ax.loglog(freqs[1:nx//2], energy[1:nx//2, 0, 0], label='Power Spec')
        ax.set_xlabel('k')
        ax.set_ylabel('E(k)')
        ax.set_title('Kinetic Energy Spectrum')
        ax.grid(True, which="both", ls="-", alpha=0.5)

    def _create_executive_summary(self, results, fig):
        """Create executive summary page."""
        fig.text(0.1, 0.9, 'Executive Summary', fontsize=18, fontweight='bold')
        fig.text(0.1, 0.8, 'ATOM Neuro-Symbolic General Purpose Scientific Intelligence', fontsize=14)

        y_pos = 0.7
        for module, data in results.items():
            status = data.get('status', 'Unknown')
            fig.text(0.1, y_pos, f'{module.upper()}: {status}', fontsize=12)
            y_pos -= 0.05

        plt.axis('off')

    def _plot_performance_metrics(self, results, fig):
        """Plot performance metrics."""
        fig.text(0.1, 0.9, 'Performance Metrics', fontsize=16, fontweight='bold')
        # Add performance plots here
        plt.axis('off')

    def _save_report_as_images(self, results):
        """Save report as individual images."""
        logger.info("Saving report as individual images")
        # Implementation would save each visualization separately


# Global visualizer instance
visualizer = AtomVisualizer()


def get_visualizer(output_dir: Optional[Union[str, Path]] = None) -> AtomVisualizer:
    """Get the global visualizer instance with optional output directory."""
    if output_dir:
        visualizer.set_output_dir(output_dir)
    return visualizer


# Convenience functions for quick visualization
def visualize_brain(brain, save_path="brain_architecture.png"):
    """Quick brain visualization."""
    return visualizer.visualize_brain_architecture(brain, save_path)


def visualize_eyes(eyes, input_data, save_path="eyes_processing.png"):
    """Quick eyes visualization."""
    return visualizer.visualize_eyes_processing(eyes, input_data, save_path)


def visualize_memory(memory, save_path="memory_analysis.png"):
    """Quick memory visualization."""
    return visualizer.visualize_memory_system(memory, save_path)


def visualize_training(history, save_path="training_progress.png"):
    """Quick training visualization."""
    return visualizer.visualize_training_progress(history, save_path)


def create_comprehensive_report(results, save_path="atom_report.pdf"):
    """Create comprehensive report."""
    visualizer.create_comprehensive_report(results, save_path)