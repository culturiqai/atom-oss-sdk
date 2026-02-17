"""
ATOM CORE: WORLD (Fluid Environment)
------------------------------------
Wraps the JAX-LBM D3Q19 Solver into a Control Environment.
Features:
- JIT-compiled Physics Loop
- Aerodynamic Reward Calculation
- FIXED: Engineering Export uses EXACT mask (No flickering solids)
"""

import jax
# CRITICAL: Enable 64-bit Precision
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from functools import partial
import time
import json 
import os

# --- DRAG REDUCTION METRICS ---
def calculate_drag_force(obs, mask):
    """
    Calculates drag force via pressure integration.
    F_d = Sum(p * n_x) where p is pressure and n_x is the x-component of the surface normal.
    For a simplified cylinder, we can approximate this via the pressure delta.
    """
    rho = obs[:, 3] # (B, X, Y, Z)
    # Average pressure (density) on windward vs leeward sides
    # Note: mask is (1, X, Y, Z)
    return jnp.mean(rho[:, 0:10]) - jnp.mean(rho[:, -10:]) 


# --- FIX FOR MACOS CRASH ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# --- PHYSICS KERNEL (D3Q19) ---
CX = jnp.array([0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0])
CY = jnp.array([0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1])
CZ = jnp.array([0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1])
W = jnp.array([1/3] + [1/18]*6 + [1/36]*12)
OPPOSITE = jnp.array([0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15])

CX_INT = [int(x) for x in CX]
CY_INT = [int(x) for x in CY]
CZ_INT = [int(x) for x in CZ]

class LBMSolver:
    def __init__(self, nx, ny, nz, precision=jnp.float64):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dtype = precision
        self.CX_b = CX.reshape(1, 19, 1, 1, 1).astype(self.dtype)
        self.CY_b = CY.reshape(1, 19, 1, 1, 1).astype(self.dtype)
        self.CZ_b = CZ.reshape(1, 19, 1, 1, 1).astype(self.dtype)
        self.W_b = W.reshape(1, 19, 1, 1, 1).astype(self.dtype)

    @partial(jax.jit, static_argnums=(0,))
    def equilibrium(self, rho, u, v, w):
        rho = jnp.expand_dims(rho, 1)
        u, v, w = jnp.expand_dims(u, 1), jnp.expand_dims(v, 1), jnp.expand_dims(w, 1)
        cu = self.CX_b*u + self.CY_b*v + self.CZ_b*w
        usq = u**2 + v**2 + w**2
        return rho * self.W_b * (1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*usq)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, f_pop, mask, u_in, tau, jet_mask=None, jet_vel=None):
        # 1. Moments
        rho = jnp.sum(f_pop, axis=1)
        inv_rho = 1.0 / (rho + 1e-15)
        ux = jnp.sum(f_pop * self.CX_b, axis=1) * inv_rho
        uy = jnp.sum(f_pop * self.CY_b, axis=1) * inv_rho
        uz = jnp.sum(f_pop * self.CZ_b, axis=1) * inv_rho

        # 2. Collision
        f_opp = f_pop[:, OPPOSITE]
        f_plus = 0.5 * (f_pop + f_opp)
        f_minus = 0.5 * (f_pop - f_opp)
        feq = self.equilibrium(rho, ux, uy, uz)
        feq_plus = 0.5 * (feq + feq[:, OPPOSITE])
        feq_minus = 0.5 * (feq - feq[:, OPPOSITE])
        omega_p = 1.0 / tau
        omega_m = 1.0 / (0.25 / (1.0/omega_p - 0.5) + 0.5)
        f_col = (f_plus - omega_p*(f_plus - feq_plus)) + (f_minus - omega_m*(f_minus - feq_minus))

        # 3. Stream
        f_stream = f_col
        for i in range(19):
            f_stream = f_stream.at[:, i].set(jnp.roll(f_col[:, i], shift=(CX_INT[i], CY_INT[i], CZ_INT[i]), axis=(1, 2, 3)))

        # 4. Boundaries
        # Inlet
        ny, nz = self.ny, self.nz
        rho_in = jnp.ones((1, ny, nz))
        u_sl = jnp.full((1, ny, nz), u_in)
        v_sl = jnp.zeros((1, ny, nz))
        w_sl = jnp.zeros((1, ny, nz))
        feq_in = self.equilibrium(rho_in[:, None], u_sl[:, None], v_sl[:, None], w_sl[:, None])
        f_stream = f_stream.at[:, :, 0].set(feq_in[:, :, 0])
        
        # Outlet
        f_stream = f_stream.at[:, :, -1].set(f_stream[:, :, -2])
        
        # Obstacle (Bounce-Back)
        mask_exp = jnp.expand_dims(mask, 1) 
        f_bounced = f_stream[:, OPPOSITE]
        f_next = f_stream * (1 - mask_exp) + f_bounced * mask_exp
        
        # Active Control (Jet Source)
        if jet_mask is not None and jet_vel is not None:
            rho_jet = jnp.ones_like(rho)
            u_j = jet_vel[:, 0, None, None, None]
            v_j = jet_vel[:, 1, None, None, None]
            w_j = jet_vel[:, 2, None, None, None]
            feq_jet = self.equilibrium(rho_jet, u_j, v_j, w_j)
            jm_exp = jnp.expand_dims(jet_mask, 1)
            f_next = f_next * (1 - jm_exp) + feq_jet * jm_exp

        # Stability Guard
        f_next = jnp.nan_to_num(f_next, nan=1.0/19.0, posinf=1.0, neginf=0.0)
        return f_next

class FluidWorld:
    def __init__(self, nx=64, ny=32, nz=32, batch_size=1):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.bs = batch_size
        self.solver = LBMSolver(nx, ny, nz)
        self.u_inlet = 0.05
        self.tau = 0.56 # Increased from 0.54 for stability at Re=1000
        self.f_state = None
        self.step_count = 0
        self.obs_radius = ny // 8
        self.base_z = nz // 2
        self.base_y = ny // 2
        self.base_x = nx // 4
        
        # MEMORY SLOT FOR THE SOLID MASK
        self.current_mask = None 

    def reset(self):
        print(">>> ATOM: Resetting World Physics...")
        rho = jnp.ones((self.bs, self.nx, self.ny, self.nz))
        u = jnp.full((self.bs, self.nx, self.ny, self.nz), self.u_inlet)
        v = jnp.zeros_like(u)
        w = jnp.zeros_like(u)
        self.f_state = self.solver.equilibrium(rho, u, v, w)
        self.step_count = 0
        
        # SAVE THE MASK
        mask = self._generate_mask(0.0)
        self.current_mask = mask
        
        return self._get_obs(self.f_state), mask

    @partial(jax.jit, static_argnums=(0,))
    def _generate_mask_batch(self, action_z_batch):
        """
        NVIDIA-GRADE FIX: Vectorized Mask Generation.
        Input: (B,) actions
        Output: (B, X, Y, Z) masks
        """
        # Define single-sample logic
        def single_mask(act):
            z_shift = (act * (self.nz / 4)).astype(jnp.int32)
            center_z = self.base_z + z_shift
            x = jnp.arange(self.nx)
            y = jnp.arange(self.ny)
            z = jnp.arange(self.nz)
            X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
            dist = (X - self.base_x)**2 + (Y - self.base_y)**2 + (Z - center_z)**2
            return (dist < self.obs_radius**2).astype(jnp.float64)

        # Vectorize over batch
        return jax.vmap(single_mask)(action_z_batch)

    def _generate_mask(self, action_z):
        # Legacy wrapper or single instance fallback
        # If action_z is scalar, expand to batch
        if np.isscalar(action_z):
            acts = jnp.full((self.bs,), action_z)
        else:
            acts = jnp.asarray(action_z)
            if acts.ndim == 0:
                acts = jnp.full((self.bs,), acts)
        
        return self._generate_mask_batch(acts)

    def step(self, action, sub_steps=20):
        # NVIDIA-GRADE FIX: Handle Batched Actions (B, 1) or (B,)
        # Ensure 'action' is a JAX array of shape (B,)
        if isinstance(action, (float, int)):
            action = jnp.full((self.bs,), action)
        else:
            action = jnp.asarray(action).reshape(-1)
            
        # Generate Batch of Masks (B, X, Y, Z) via vmap
        mask = self._generate_mask_batch(action)
        self.current_mask = mask
        
        def loop_body(f, _):
            return self.solver.step(f, mask, self.u_inlet, self.tau), None

        self.f_state, _ = jax.lax.scan(loop_body, self.f_state, jnp.arange(sub_steps))
        self.step_count += sub_steps
        
        obs = self._get_obs(self.f_state)
        
        # --- PHYSICS-BASED REWARD ---
        reward = self._compute_reward(self.f_state, mask, obs)
        
        return obs, reward, False, {"mask": mask}
    
    def _compute_reward(self, f_state, mask, obs):
        """
        NVIDIA-GRADE FIX: Per-Sample Reward Calculation.
        Returns: (B, 1) Numpy Array (on CPU) to avoid blocking the training loop.
        """
        ux = obs[:, 0]  # (B, X, Y, Z)
        rho = obs[:, 3]
        
        # Pressure at inlet vs outlet (proxy for drag)
        # MEAN over spatial dims (1,2,3), keep Batch (0)
        p_inlet = jnp.mean(rho[:, 2:5, :, :], axis=(1,2,3))    # Near inlet
        p_outlet = jnp.mean(rho[:, -5:-2, :, :], axis=(1,2,3)) # Near outlet
        pressure_drop = (p_inlet - p_outlet) * 10.0
        
        # Wake deficit
        u_inlet_actual = jnp.mean(ux[:, 2:5, :, :], axis=(1,2,3))
        u_outlet_actual = jnp.mean(ux[:, -5:-2, :, :], axis=(1,2,3))
        wake_deficit = (u_inlet_actual - u_outlet_actual) * 50.0
        
        # Combined drag metric
        drag = pressure_drop + wake_deficit
        
        # 2. ENSTROPHY (Energy Dissipation Penalty)
        uy = obs[:, 1]
        uz = obs[:, 2]
        
        # Simple gradient approximation (du/dy, du/dz, etc.)
        du_dy = jnp.diff(ux, axis=2) # Axis 2 is Y
        dv_dx = jnp.diff(uy, axis=1) # Axis 1 is X
        
        # Vorticity magnitude
        # We must align shapes since diff reduces dim size
        # Quick robust metric: just mean squared sum
        vort_z = jnp.mean(du_dy**2, axis=(1,2,3)) + jnp.mean(dv_dx**2, axis=(1,2,3))
        enstrophy = vort_z * 100.0
        
        # 3. COMBINED REWARD (B,)
        reward = -drag - 0.1 * enstrophy
        
        # Expand to (B, 1) and convert to Numpy CPU list immediately
        return np.array(reward).reshape(-1, 1)

    def _get_obs(self, f):
        rho = jnp.sum(f, axis=1)
        inv_rho = 1.0 / (rho + 1e-15)
        ux = jnp.sum(f * self.solver.CX_b, axis=1) * inv_rho
        uy = jnp.sum(f * self.solver.CY_b, axis=1) * inv_rho
        uz = jnp.sum(f * self.solver.CZ_b, axis=1) * inv_rho
        return jnp.stack([ux, uy, uz, rho], axis=1) 

    def export_to_web(self, filename="volumetric_data.json"):
        try:
            obs = self._get_obs(self.f_state)
            ux = np.array(obs[0, 0])
            uy = np.array(obs[0, 1])
            uz = np.array(obs[0, 2])
            speed = np.sqrt(ux**2 + uy**2 + uz**2)

            if self.current_mask is not None:
                real_mask = np.array(self.current_mask[0])
            else:
                real_mask = np.zeros_like(speed)

            step = 2 
            x_range = np.arange(0, self.nx, step)
            y_range = np.arange(0, self.ny, step)
            z_range = np.arange(0, self.nz, step)
            X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
            
            ux_s = ux[0:self.nx:step, 0:self.ny:step, 0:self.nz:step]
            uy_s = uy[0:self.nx:step, 0:self.ny:step, 0:self.nz:step]
            uz_s = uz[0:self.nx:step, 0:self.ny:step, 0:self.nz:step]
            speed_s = speed[0:self.nx:step, 0:self.ny:step, 0:self.nz:step]
            mask_s = real_mask[0:self.nx:step, 0:self.ny:step, 0:self.nz:step]
            
            save_mask = (mask_s > 0.5) | (speed_s > 0.01)
            
            X_flat = X[save_mask]
            Y_flat = Y[save_mask]
            Z_flat = Z[save_mask]
            ux_flat = ux_s[save_mask]
            uy_flat = uy_s[save_mask]
            uz_flat = uz_s[save_mask]
            solid_flat = (mask_s[save_mask] > 0.5).astype(int)
            speed_flat = speed_s[save_mask]
            
            data_stack = np.stack([
                X_flat, Y_flat, Z_flat, 
                ux_flat, uy_flat, uz_flat, 
                solid_flat, speed_flat
            ], axis=1)
            
            data_stack[:, 3:6] = np.round(data_stack[:, 3:6], 3)
            data_stack[:, 7] = np.round(data_stack[:, 7], 3)
            
            packed_data = data_stack.flatten().tolist()

            temp_name = filename + ".tmp"
            with open(temp_name, 'w') as f:
                json.dump(packed_data, f)
            os.replace(temp_name, filename)
        except Exception as e:
            print(f"Export failed: {e}")

    def render(self, filename="ignored"):
        self.export_to_web("volumetric_data.json")

class CylinderWorld(FluidWorld):
    """
    Real-World Aerodynamic Challenge: Active Flow Control.
    Constraint: Fixed cylinder with dual variable-velocity jets on the trailing edge.
    Objective: Minimize lift fluctuation (vortex shedding) and drag.
    """
    def __init__(self, nx=64, ny=32, nz=32, batch_size=1):
        super().__init__(nx, ny, nz, batch_size)
        self.cylinder_radius = ny // 6
        self.jet_radius = 2
        self.jet_offset = self.cylinder_radius + 1
        
        # Fixed Cylinder Location
        self.cx, self.cy, self.cz = nx // 4, ny // 2, nz // 2
        
        # Pre-generate static cylinder mask
        x = jnp.arange(self.nx)
        y = jnp.arange(self.ny)
        z = jnp.arange(self.nz)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        
        # Cylinder aligned with Z-axis (spans full height)
        dist_cyl = (X - self.cx)**2 + (Y - self.cy)**2
        self.static_mask = (dist_cyl < self.cylinder_radius**2).astype(jnp.float64)
        self.static_mask = jnp.expand_dims(self.static_mask, 0) # (1, nx, ny, nz)
        
        # Jet Positions (Top and Bottom of trailing edge)
        self.jet_mask_top = ((dist_cyl < (self.cylinder_radius + 2)**2) & 
                            (dist_cyl > self.cylinder_radius**2) &
                            (Y > self.cy + self.jet_offset - 2) &
                            (X > self.cx) & (X < self.cx + 5)).astype(jnp.float64)
        
        self.jet_mask_bot = ((dist_cyl < (self.cylinder_radius + 2)**2) & 
                            (dist_cyl > self.cylinder_radius**2) &
                            (Y < self.cy - self.jet_offset + 2) &
                            (X > self.cx) & (X < self.cx + 5)).astype(jnp.float64)
        
        self.jet_mask = jnp.expand_dims(self.jet_mask_top + self.jet_mask_bot, 0)
        self.current_mask = self.static_mask

    def reset(self):
        print(">>> ATOM: Resetting Cylinder Challenge...")
        rho = jnp.ones((self.bs, self.nx, self.ny, self.nz))
        u = jnp.full((self.bs, self.nx, self.ny, self.nz), self.u_inlet)
        v = jnp.zeros_like(u)
        w = jnp.zeros_like(u)
        self.f_state = self.solver.equilibrium(rho, u, v, w)
        self.step_count = 0
        self.current_mask = self.static_mask
        return self._get_obs(self.f_state), self.static_mask

    def step(self, action, sub_steps=20):
        # NVIDIA-GRADE FIX: Batched Action Handling
        if isinstance(action, (float, int)):
            # Scalar case
            v_act = jnp.full((self.bs,), action)
        else:
            # Batch case (B, 1) or (B,)
            v_act = jnp.asarray(action).reshape(self.bs)
            
        # Construct Jet Velocity Batch (B, 3)
        # u = u_inlet (fixed), v = action * 0.1, w = 0.0
        u_js = jnp.full((self.bs,), self.u_inlet)
        v_js = v_act * 0.1
        w_js = jnp.zeros((self.bs,))
        
        jet_velocity = jnp.stack([u_js, v_js, w_js], axis=1) # (B, 3)
        
        def loop_body(f, _):
            return self.solver.step(f, self.static_mask, self.u_inlet, self.tau, 
                                   jet_mask=self.jet_mask, jet_vel=jet_velocity), None

        self.f_state, _ = jax.lax.scan(loop_body, self.f_state, jnp.arange(sub_steps))
        self.step_count += sub_steps
        obs = self._get_obs(self.f_state)
        reward, cd = self._compute_cylinder_reward(obs)
        return obs, reward, False, {"mask": self.static_mask, "lift": reward, "cd": cd}

    def _compute_cylinder_reward(self, obs):
        ux, uy = obs[:, 0], obs[:, 1]
        
        # Wake fluctuation (proxy for vortex lift)
        # MEAN over spatial, keep Batch
        wake_uy = jnp.mean(jnp.abs(uy[:, self.cx + self.cylinder_radius : self.cx + 20]), axis=(1,2,3))
        lift_penalty = -wake_uy * 10.0
        
        # Drag proxy (velocity delta)
        u_in = jnp.mean(ux[:, 2:5], axis=(1,2,3))
        u_out = jnp.mean(ux[:, -10:-2], axis=(1,2,3))
        drag_penalty = -jnp.abs(u_in - u_out) * 5.0
        
        # Total reward (B,)
        reward = lift_penalty + drag_penalty
        
        # Cd Proxy (B,)
        cd_proxy = jnp.abs(u_in - u_out) * 10.0
        cd_proxy = jnp.nan_to_num(cd_proxy, nan=0.0)

        # NaN safety
        reward = jnp.nan_to_num(reward, nan=0.0)
        
        # Return (B, 1) Numpy
        return np.array(reward).reshape(-1, 1), np.array(cd_proxy).reshape(-1, 1)

class MeshWorld(FluidWorld):
    """
    Custom Geometry World: 'God Mode' for User Experiments.
    Loads any .stl file, voxelizes it, and runs flow simulation over it.
    """
    def __init__(self, stl_path: str, nx=128, ny=64, nz=64, batch_size=1):
        super().__init__(nx, ny, nz, batch_size)
        self.stl_path = stl_path
        
        # Load and Voxelize Mesh
        try:
            import trimesh
        except ImportError:
            raise ImportError("Please install trimesh: pip install trimesh")
            
        if not os.path.exists(stl_path):
            raise FileNotFoundError(f"STL file not found: {stl_path}")
            
        print(f">>> [ATOM] Voxelizing {stl_path} for Grid {nx}x{ny}x{nz}...")
        mesh = trimesh.load(stl_path)
        
        # 1. Normalize Scale to Fit Grid (occupy 30% of length)
        target_len = nx * 0.30
        current_len = mesh.extents[0]
        scale = target_len / (current_len + 1e-6)
        mesh.apply_scale(scale)
        
        # 2. Center in Flow Domain (1/3 downstream, centered Y/Z)
        mesh_center = mesh.centroid
        target_center = np.array([nx // 3, ny // 2, nz // 2])
        translation = target_center - mesh_center
        mesh.apply_translation(translation)
        
        # 3. Voxelize
        # Pitch = 1.0 (1 unit = 1 lattice node)
        voxel_grid = mesh.voxelized(pitch=1.0)
        
        # 4. Fill 3D Mask
        indices = voxel_grid.points.astype(int)
        
        # Filter bounds
        valid_mask = (
            (indices[:, 0] >= 0) & (indices[:, 0] < nx) &
            (indices[:, 1] >= 0) & (indices[:, 1] < ny) &
            (indices[:, 2] >= 0) & (indices[:, 2] < nz)
        )
        indices = indices[valid_mask]
        
        # Create Boolean Mask
        self.static_mask_np = np.zeros((nx, ny, nz), dtype=bool)
        self.static_mask_np[indices[:, 0], indices[:, 1], indices[:, 2]] = True
        
        # Convert to JAX
        self.static_mask = jnp.array(self.static_mask_np, dtype=jnp.float64)
        self.static_mask = jnp.expand_dims(self.static_mask, 0) # (1, X, Y, Z)
        
        self.current_mask = self.static_mask
        print(f">>> [ATOM] Mesh Voxelized. Solid Nodes: {len(indices)}")

    def reset(self):
        print(f">>> ATOM: Resetting Flow over {os.path.basename(self.stl_path)}...")
        rho = jnp.ones((self.bs, self.nx, self.ny, self.nz))
        u = jnp.full((self.bs, self.nx, self.ny, self.nz), self.u_inlet)
        v = jnp.zeros_like(u)
        w = jnp.zeros_like(u)
        self.f_state = self.solver.equilibrium(rho, u, v, w)
        self.step_count = 0
        self.current_mask = self.static_mask
        return self._get_obs(self.f_state), self.static_mask

    def step(self, action, sub_steps=20):
        # ACTIVE CONTROL: "Virtual Wind Tunnel"
        # Action controls the Inlet Vector (y, z components) to steer flow.
        
        # NVIDIA-GRADE FIX: Batched Actions (B, 2)
        # We expect action to be (B, 2) or (2,) for single
        
        # Ensure array
        acts = jnp.asarray(action)
        if acts.ndim == 1:
             # Case: Single sample (2,) -> Expand to (B, 2)
             # OR Case: Batch of scalars? No, MeshWorld expects 2 dims per sample.
             # If B=1, (2,) is fine. If B>1, (B*2,) flattened is ambiguous.
             # We assume (B, 2) if B > 1.
             if self.bs == 1:
                 acts = acts.reshape(1, -1)
             else:
                 acts = acts.reshape(self.bs, -1)
        
        # acts is (B, 2)
        # vy_ctrl = acts[:, 0] * 0.05
        # vz_ctrl = acts[:, 1] * 0.05
        
        vy_ctrl = acts[:, 0] * 0.05
        vz_ctrl = acts[:, 1] * 0.05
        
        # Construct Jet Velocity Batch (B, 3)
        # u = u_inlet, v = vy, w = vz
        u_js = jnp.full((self.bs,), self.u_inlet)
        
        jet_vel = jnp.stack([u_js, vy_ctrl, vz_ctrl], axis=1) # (B, 3)
        
        # Create a "Wind Tunnel Nozzle" mask at x=0
        # (This is a simplified approach reusing the jet logic)
        nozzle_mask = np.zeros((1, self.nx, self.ny, self.nz))
        nozzle_mask[:, 0:2, :, :] = 1.0 # First 2 layers
        nozzle_mask = jnp.array(nozzle_mask)
        
        def loop_body(f, _):
            # Using jet_mask to override inlet velocity
            return self.solver.step(f, self.static_mask, self.u_inlet, self.tau,
                                   jet_mask=nozzle_mask, jet_vel=jet_vel), None

        self.f_state, _ = jax.lax.scan(loop_body, self.f_state, jnp.arange(sub_steps))
        self.step_count += sub_steps
        
        obs = self._get_obs(self.f_state)
        
        # Reward: Minimize Drag, Minimize Turbulence (Enstrophy)
        # We want the agent to find the "Streamline Angle" for this object
        # Uses Base Class _compute_reward which is already Vectorized (B, 1) output
        reward = self._compute_reward(self.f_state, self.static_mask, obs)
        
        # reward is (B, 1) numpy array
        return obs, reward, False, {"mask": self.static_mask, "drag": -reward}

if __name__ == "__main__":
    env = CylinderWorld(nx=64, ny=32, nz=32)
    env.reset()
    for i in range(10):
        o, r, d, info = env.step([0.1])
        print(f"Step {i}: Reward {r:.4f}")
        env.export_to_web()
    print("Challenge Complete.")