import numpy as np
import trimesh as tm

from mesh_generator import generate_mesh

def effects(mesh, n=0):
    mesh = mesh.copy()

    def rotate(mesh, n):
        rotation_matrix_x = tm.transformations.rotation_matrix(np.radians(1.0*n+30), [1,0,0], point=mesh.centroid)
        rotation_matrix_y = tm.transformations.rotation_matrix(np.radians(1.0*n+10), [0,1,0], point=mesh.centroid)
        rotation_matrix_z = tm.transformations.rotation_matrix(np.radians(1.0*n+0), [0,0,1], point=mesh.centroid)

        mesh.vertices = tm.transformations.transform_points(mesh.vertices, rotation_matrix_x)
        mesh.vertices = tm.transformations.transform_points(mesh.vertices, rotation_matrix_y)
        mesh.vertices = tm.transformations.transform_points(mesh.vertices, rotation_matrix_z)

        return mesh


    return rotate(mesh, n)


if __name__ == '__main__':
    import time

    mesh, mesh_name = generate_mesh()

    # Measure time
    total_time = 0
    for i in range(100):
        start_time = time.time()
        mesh = effects(mesh)
        total_time += time.time() - start_time
    print(f"Execution time: {total_time/100:.4f} seconds")
    print(f'Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}, Unique edges: {len(mesh.edges)}, Components: {len(tm.graph.connected_components(mesh.edges))}')
    print(mesh_name)

    try:
        import pyvista as pv

        assert mesh.faces.shape[1] == 3
        mesh = pv.PolyData(mesh.vertices, np.hstack((np.full((mesh.faces.shape[0], 1), 3), mesh.faces)))

        pl = pv.Plotter()
        # pl.enable_parallel_projection()
        # Show virtual display in the xy plane
        virtual_display = pv.Rectangle().translate([-0.5,-0.5,0], inplace = True)
        pl.add_mesh(virtual_display, style = 'wireframe')

        pl.add_mesh(mesh, show_edges = True, opacity = 0.5)
        pl.add_points(mesh.points, color='blue', point_size=3)
        pl.camera_position = [(0, 0, 6), (0, 0, 2), (0, 1, 0)]
        pl.show()
    except Exception as e: print(e)