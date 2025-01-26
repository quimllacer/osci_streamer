import numpy as np
from collections import defaultdict, deque

import time

def construct_graph(edges):
    """Constructs a graph from edges."""
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    return graph

def find_odd_degree_vertices(graph):
    """Finds vertices of odd degree in the graph."""
    return np.array([v for v, adj in graph.items() if len(adj) % 2 != 0])

def ensure_connectivity(edges, graph):
    """Ensure graph connectivity by identifying and connecting disconnected components."""
    visited = set()
    
    def dfs(v):
        visited.add(v)
        for neighbor in graph[v]:
            if neighbor not in visited:
                dfs(neighbor)
    
    components = []
    for vertex in graph:
        if vertex not in visited:
            dfs(vertex)
            components.append(vertex)
    
    # Connect disconnected components by adding an edge between them
    new_edges_to_add = np.array([(components[i-1], components[i]) for i in range(1, len(components))], dtype=int)
    if new_edges_to_add.size > 0:
        edges = np.vstack([edges, new_edges_to_add])
    
    return edges

def duplicate_graph_values(graph):
    """Duplicates the adjacency lists for each vertex in the graph."""
    for vertex, neighbors in graph.items():
        graph[vertex].extend(neighbors)  # Duplicates the values in-place
        graph[vertex].sort() # Try removing this, it affects the path behaviour over duplicated edges.
    return graph

def find_eulerian_path(edges, graph):
    edges = ensure_connectivity(edges, graph)
    graph = construct_graph(edges) # Correct redundancy
    odd_degree_vertices = find_odd_degree_vertices(graph)

    # Check if eulerian path exists. If not, duplicate all edges to force it.
    if len(odd_degree_vertices) not in [0, 2]:
        edges = np.vstack([edges, edges])
        duplicate_graph_values(graph)
        # print('An Eulerian path does not exist. Solved by duplicating all edges.')

    if len(odd_degree_vertices) == 2:
        start_vertex = odd_degree_vertices[0]
    else:
        start_vertex = next(iter(graph))

    # Iterative DFS to find an Eulerian path or cycle
    def dfs_iterative(start_vertex):
        """Finds an Eulerian path or cycle using iterative DFS efficiently."""
        stack = [start_vertex]
        path = deque()  # Use deque for efficient appends at the end and reverse at the end
        while stack:
            vertex = stack[-1]
            if graph[vertex]:
                # Pop efficiently from the right of the adjacency list
                next_vertex = graph[vertex].pop()
                # Use a set for adjacency to avoid the costly remove operation
                graph[next_vertex].remove(vertex)  
                stack.append(next_vertex)
            else:
                path.appendleft(stack.pop())  # Reverse the collection on the fly
        return path

    path = np.array(dfs_iterative(start_vertex))
    return path

def path_length(vertices, path):
    vartices = vertices[path]
    # Convert the list of vertices to a NumPy array
    vertices_array = np.array(vertices)
    # Calculate the differences between consecutive vertices
    diffs = np.diff(vertices_array, axis=0)
    # Calculate the Euclidean distance for each segment and sum them up
    total_length = np.sum(np.linalg.norm(diffs, axis=1))
    return total_length

def interpolate_path(vertices, path, t):

    # Calculate total number of segments and scale t to the total number of segments
    total_segments = len(path) - 1
    scaled_t = t * total_segments
    segment_index = np.floor(scaled_t).astype(int)
    
    # Calculate local t within the current segments
    local_t = scaled_t - segment_index
    
    # Clip segment_index to ensure it's within bounds
    segment_index = np.clip(segment_index, 0, total_segments - 1)
    
    # Linearly interpolate between the start and end vertices for each t
    start_vertices = vertices[path[segment_index]]
    end_vertices = vertices[path[segment_index + 1]]
    interpolated_vertices = start_vertices + (end_vertices - start_vertices) * local_t[:, None]
    
    return interpolated_vertices

def save_data(vertices, path, filename):
    # Prepare the data as a dictionary
    data = {
        'vertices': vertices,
        'path': path
    }
    # Replace or append a custom extension for the data file
    data_filename = filename.replace('.obj', '.data')
    # Serialize and save the data to a file
    with open(data_filename, 'wb') as file:
        pickle.dump(data, file)

def load_data(directory_path, filename):
    # Replace or append the custom extension to match the save format
    data_filename = os.path.join(directory_path, filename)
    data_filename = data_filename.replace('.obj', '.data')
    # Deserialize the data from the file
    with open(data_filename, 'rb') as file:
        data = pickle.load(file)
    # Return the loaded vertices and path
    return data['vertices'], data['path']

def process_files(vertices, edges):

                print(f'Vertices count: {len(vertices)}')
                print(f'Unique edges count: {len(edges)}')

                path = find_eulerian_path(edges, construct_graph(edges))
                print(f'Eulerian path vertices: {len(path)}')
                length = path_length(vertices, path)
                print(f'Eulerian path distance: {int(length)}')

                return path

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import trimesh as tm

    from mesh_generator import generate_mesh
    from effects import effects
    from mesh_to_edges import mesh_to_edges
    
    mesh, mesh_name = generate_mesh()
    mesh = effects(mesh)
    # projected_mesh, edges = mesh_to_edges(mesh)
    process_files(mesh.vertices, mesh.edges_unique)

    t = np.linspace(0, 1, 512*3, endpoint = False)

    path = find_eulerian_path(mesh.edges_unique, construct_graph(mesh.edges_unique))

    # Measure time
    total_time = 0
    for i in range(100):
        start_time = time.time()
        interpath = interpolate_path(mesh.vertices, path, t)

        total_time += time.time() - start_time
    print(f"Execution time: {total_time/100:.6f} seconds")

    
    plt.plot(interpath[:,1], -interpath[:,0])
    plt.show()


