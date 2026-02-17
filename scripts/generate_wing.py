import numpy as np
import trimesh

def naca4_symmetric(code, n_points=100, chord=1.0):
    """Generate NACA 4-digit symmetric airfoil coordinates."""
    t = int(code[-2:]) / 100.0
    x = np.linspace(0, chord, n_points)
    yt = 5 * t * 0.2 * (0.2969 * np.sqrt(x/chord) - 0.1260 * (x/chord) - 
                        0.3516 * (x/chord)**2 + 0.2843 * (x/chord)**3 - 
                        0.1015 * (x/chord)**4)
    x_upper = x
    y_upper = yt
    x_lower = x
    y_lower = -yt
    
    # Concatenate to form loop
    x_coor = np.concatenate([x_upper[::-1], x_lower[1:]])
    y_coor = np.concatenate([y_upper[::-1], y_lower[1:]])
    return np.column_stack([x_coor, y_coor])

def extrude_wing(profile, span=2.0):
    """Extrude 2D profile into 3D wing."""
    # Create 3D points
    n_pts = len(profile)
    points_root = np.column_stack([profile, np.zeros(n_pts)])
    points_tip = np.column_stack([profile, np.full(n_pts, span)])
    
    # Create visualization cloud
    points = np.vstack([points_root, points_tip])
    
    # Triangulate (simple strip)
    faces = []
    for i in range(n_pts - 1):
        # Two triangles per segment
        # Root[i], Root[i+1], Tip[i]
        faces.append([i, i+1, i+n_pts])
        # Tip[i], Root[i+1], Tip[i+1]
        faces.append([i+n_pts, i+1, i+n_pts+1])
        
    # Close ends (Cap) - Simplified for robustness (just a convex hull if needed, or simple fan)
    # For now, let's just make it a watertight hull via Trimesh if possible or leave open
    # Trimesh can fix normals
    
    mesh = trimesh.Trimesh(vertices=points, faces=faces)
    return mesh

def main():
    print("Generating Test Wing (NACA 0012)...")
    profile = naca4_symmetric("0012")
    mesh = extrude_wing(profile, span=0.5)
    
    # Post-process to ensure watertightness for voxelizer
    # Using convex hull for guaranteed solid water-tightness in this demo
    mesh = mesh.convex_hull 
    
    output = "test_wing.stl"
    mesh.export(output)
    print(f"Saved to {output}")

if __name__ == "__main__":
    main()
