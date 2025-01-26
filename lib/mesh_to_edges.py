import numpy as np
import trimesh as tm

def mesh_to_edges(mesh):
    mesh = mesh.copy()

    # Perspective transformation
    if True:
        camera_z_position=-6
        focal_point=2


        points_homogeneous = np.hstack([mesh.vertices, np.ones((mesh.vertices.shape[0], 1))])
        
        # Define the perspective projection matrix (4x4)
        projection_matrix = np.array([
            [focal_point, 0, 0, 0],
            [0, focal_point, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1/camera_z_position, 1]
        ])
        
        # Apply the projection matrix to the points
        projected_points_homogeneous = points_homogeneous @ projection_matrix.T
        
        # Perform perspective division to get 2D points
        w = projected_points_homogeneous[:, 3]  # Extract the w coordinate
        mesh.vertices = projected_points_homogeneous[:, :3] / w[:, np.newaxis]

    # 3D back-face culling
    if True:
        mesh.faces = mesh.faces[mesh.face_normals[:, 2] > 0.0]

    # Extract boundary and feature edges
    if False:
        pass

    # Remove edge between adjacent coplanar faces
    if True:
        # Define a threshold for face normal similarity (coplanarity tolerance)
        threshold = 0.9999

        # Get face normals and adjacency list
        normals = mesh.face_normals
        adjacent_faces = mesh.face_adjacency

        # Use set to store adjacent coplanar edges
        adj_cop_edges = set()

        # Iterate through adjacent faces
        for pair in adjacent_faces:
            normal1 = normals[pair[0]]
            normal2 = normals[pair[1]]

            # Calculate dot product to check for coplanarity
            dot_product = np.dot(normal1, normal2)

            if dot_product > threshold:
                # If coplanar, find the shared edge between the two faces
                shared_edge = set(mesh.faces[pair[0]]).intersection(mesh.faces[pair[1]])
                if len(shared_edge) == 2:  # Ensure it's an edge (2 vertices)
                    adj_cop_edges.add(tuple(sorted(shared_edge)))  # Sort to avoid duplicate edges

        # Get unique edges in the mesh
        unique_edges = set(map(tuple, map(sorted, mesh.edges_unique)))

        # Subtract coplanar edges from all unique edges
        non_adj_cop_edges = unique_edges - adj_cop_edges

    # Ensure non_adj_cop_edges exists even if no coplanar edges are found
    if 'non_adj_cop_edges' not in vars():
        non_adj_cop_edges = mesh.edges_unique
    else:
        # Convert the set back to a NumPy array
        non_adj_cop_edges = np.array(list(non_adj_cop_edges))



    # Project on the xy plane
    if True:
        # Add the Z=0 coordinate to get the 3D output
        mesh.vertices[:, 2] = 0

    return mesh, non_adj_cop_edges

    
if __name__ == '__main__':
    import time
    from mesh_generator import generate_mesh
    from effects import effects

    mesh, mesh_name = generate_mesh()
    mesh = effects(mesh)


    # Measure time
    total_time = 0
    for i in range(100):
        start_time = time.time()
        projected_mesh, edges = mesh_to_edges(mesh)
        total_time += time.time() - start_time
    print(f"Execution time: {total_time/100:.4f} seconds")
    print(f'Vertices: {len(projected_mesh.vertices)}, Faces: {len(projected_mesh.faces)}, Unique edges: {len(edges)}, Components: {len(tm.graph.connected_components(edges))}')
    print(mesh_name)

    try:
        import pyvista as pv

        print(f'Number of visible lines: {len(edges)}')

        assert edges.shape[1] == 2
        lines = np.hstack((np.full((edges.shape[0], 1), 2), edges))
        projected_mesh = pv.PolyData(projected_mesh.vertices, lines = lines)

        assert mesh.faces.shape[1] == 3
        faces = np.hstack((np.full((mesh.faces.shape[0], 1), 3), mesh.faces))
        mesh = pv.PolyData(mesh.vertices, faces = faces)

        pl = pv.Plotter()
        # Show virtual display in the xy plane
        virtual_display = pv.Rectangle().translate([-0.5,-0.5,0], inplace = True)
        pl.add_mesh(virtual_display, style = 'wireframe')

        pl.add_mesh(mesh, show_edges = True, opacity = 0.5)
        pl.add_mesh(projected_mesh, color = 'red', line_width = 3)
        pl.add_points(mesh.points, color='blue', point_size=3)
        pl.camera_position = [(0, 0, 6), (0, 0, 2), (0, 1, 0)]
        pl.show()
    except Exception as e: print(e)