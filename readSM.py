import random

def read_graphs_from_file(file_path):
    graphs = {}
    current_graph_id = None
    current_graph = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if parts[0] == 't':
                counter=0
                if len(current_graph) <= 24:  # If graph size is smaller than or equal to 16
                        graphs[current_graph_id] = None  # Add None instead of the graph
                else:
                        graphs[current_graph_id] = current_graph
                current_graph_id = int(parts[2])
                current_graph = {}
            elif parts[0] == 'v':
                vertex_id = int(parts[1])
                current_graph[vertex_id] = set()
                counter+=1
            elif parts[0] == 'e':
                #max=vertex_id
                vertex1 = int(parts[1])
                vertex2 = int(parts[2])
                current_graph[vertex1].add(vertex2)
                current_graph[vertex2].add(vertex1)
        
        if current_graph_id is not None:
            if (counter<=24):
                graphs[current_graph_id] =None
                print("no",graphs[current_graph_id])
            else:
                graphs[current_graph_id] = current_graph
            
    return graphs

def write_graph_to_file(graph, output_file_path):
    with open(output_file_path, 'w') as file:
        for vertex, neighbors in graph.items():
            for neighbor in neighbors:
                if vertex < neighbor:  # Ensure each edge is written once
                    file.write(f"{vertex} {neighbor}\n")

def main():
    input_file_path = 'Dataset_SM/AIDS.txt'
    output_file_path = 'Dataset_SM/AIDS_D'
    suffix=".txt"
    target_graph_id = 1  # Set the target graph ID here
    random_numbers = random.sample(range(0, 40000), 3000)
    graphs = read_graphs_from_file(input_file_path)
    counter=0
    for target_graph_id in random_numbers:
        if graphs[target_graph_id] is not None:
            if counter<2000:
                write_graph_to_file(graphs[target_graph_id], output_file_path+str(counter)+suffix)
            #print(f"Graph {target_graph_id} written to {output_file_path}")
                counter+=1
        else:
            print(f"Graph {target_graph_id} not found in the input file.")

if __name__ == "__main__":
    main()
