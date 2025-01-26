import os
import numpy as np
import trimesh as tm
import time

os.chdir(os.path.dirname(__file__))

def generate_mesh(idx = 6, timer = 0, **kwargs):
    meshes = {
        # 'Toroid': tm.creation.torus,
        # 'Icosphere': tm.creation.icosphere,
        'Uv_sphere': tm.creation.uv_sphere,
        'Capsule': tm.creation.capsule,
        # 'Cone': tm.creation.cone,
        'Cylinder': tm.creation.cylinder,
        # Platonic solids
        'Cube': tm.creation.box,
        # 'Icosahedron': tm.creation.icosphere,
        #'Load_monkey': tm.exchange.load.load_mesh,
        'Load_octahedron': tm.exchange.load.load_mesh,
        'Load_torus': tm.exchange.load.load_mesh,
    }

    # Function-specific arguments
    parametric_resolution = 20
    kwargs = {
        # 'Toroid': {'major_radius':1, 'minor_radius':0.3, 'major_sections':16, 'minor_sections':16,},
        # 'Icosphere': {'subdivisions':1, 'radius':1,},
        'Uv_sphere': {'radius':1, 'count': [10, 10]},
        'Capsule': {'height':4, 'radius':1, 'count':[10, 10],},
        # 'Cone': {'radius':1, 'height':1, 'sections':32,},
        'Cylinder': {'radius':1, 'height':1, 'sections':16*4,},
        # Platonic solids
        'Cube': {},
        # 'Icosahedron': {},
        # Load external
        # 'Load_monkey': {'file_obj': '../files_to_render/monkey.obj',
        #              'process':False,
        #              },
        'Load_octahedron': {'file_obj': '../assets/objects/octahedron.obj',
                     'process':False,
                     },
        'Load_torus': {'file_obj': '../assets/objects/torus.obj',
                     'process':False,
                     },



    }
    def call_mesh(mesh, **kwargs):
        return mesh(**kwargs)

    # Select mesh based on the current timer
    idx = int(idx % len(meshes))
    mesh_name, mesh = list(meshes.items())[idx]
    
    # Call function with or without args
    mesh = call_mesh(mesh, **kwargs.get(mesh_name, ()))

    # Cleans the mesh of repeated points, etc.
    mesh.process(True, merge_norm=True, merge_tex=True)
    # Normalize mesh size
    mesh_bounds = np.reshape(mesh.bounds, (3,2))
    norm_factor = max(mesh_bounds[:,1]-mesh_bounds[:,0])
    scale_and_transform = tm.transformations.scale_and_translate(1/norm_factor, [0,0,2])
    mesh.vertices = tm.transformations.transform_points(mesh.vertices, scale_and_transform)


    return mesh, mesh_name


if __name__ == '__main__':
    # Measure time
    total_time = 0
    for i in range(100):
        start_time = time.time()
        mesh, mesh_name = generate_mesh()
        total_time += time.time() - start_time
    print(f"Execution time: {total_time/100:.4f} seconds")
    print(f'Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}, Unique edges: {len(mesh.edges_unique)}, Components: {len(tm.graph.connected_components(mesh.edges))}')
    
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