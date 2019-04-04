#@title #Visualize example graphs  { form-width: "30%" }
def rgb_from_hex(h):
    if h[0] != '#' or len(h) != 7:
        raise ValueError("'{}' should be '#' followed by 6 HEX chars".format(h))
    h = h.lstrip('#')
    h = h.upper()
    invalid_chars = set(list(h)) - set(list("0123456789ABCDEF"))
    if invalid_chars:
        raise ValueError("Incorrect chars: \{{}\}".format(', '.format(invalid_chars)))
    return tuple(int(h[i:i+2], 16)/255.0 for i in [0, 2 ,4])

#@markdown ##General Vizual Params

node_size=200 #@param{type:"slider", min:128, max:2048, step:1}
node_hex_color = "#808080" #@param {type:"string"}
node_color = rgb_from_hex(node_hex_color)
node_linewidth=1.0 #@param{type:"slider", min:0.1, max:3.0, step:0.1}
edge_width=0.2 #@param{type:"slider", min:0.1, max:3.0, step:0.1}
edge_style = "dashed" #@param ["solid", "dashed", "dotted", "dashdot"]
start_color="w"
end_color="k"

#@markdown ##Solution Vizual Paramters
solution_node_hex_color = "#3DFF3D" #@param {type:"string"}
solution_node_color = rgb_from_hex(solution_node_hex_color)
solution_node_linewidth=0.6 #@param{type:"slider", min:0.1, max:6.0, step:0.1}
solution_edge_width=6.0 #@param{type:"slider", min:0.1, max:4.0, step=0.1}
solution_edge_style = "solid" #@param ["solid", "dashed", "dotted", "dashdot"]


#@markdown ##Specific Parameters

seed = 5  #@param{type: 'integer'}
rand = np.random.RandomState(seed=seed)


num_examples = 2  #@param{type: 'integer'}


min_nodes = 4 #@param {type:"slider", min:4, max:64, step:1}
max_nodes = 5 #@param {type:"slider", min:4, max:64, step:1}

min_max_nodes = (min_nodes, max_nodes)

graphs = generate_raw_graphs(
    rand,
    num_examples,
    min_max_nodes,
)