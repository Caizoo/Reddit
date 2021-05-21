import pandas as pd 
import json 
from graph_tool.all import * 

def draw_graph(gr, pos, nodes_attr, edges_attr, save_graph, o_size):
    o = 'graph.png' if save_graph else None
    graph_draw(gr, pos, vertex_text=nodes_attr['text'], edge_pen_width=edges_attr, 
                fit_view=True, output_size=o_size, output=o,
                vertex_fill_color=nodes_attr['color'], vertex_text_color=nodes_attr['text_color'])

def print_graph(df: pd.DataFrame, sub: str='', number_subs: bool=False, self_loops: bool=False, no_loop_other: bool=False, 
                inc_non_loops: bool=False, save_graph: bool=False):
    
    gr = Graph(directed=True) 
    nodes = {

    }
    nodes_attr = {
        "text": gr.new_vertex_property("object"),
        "color": gr.new_vertex_property("vector<double>"),
        "text_color": gr.new_vertex_property("vector<double>"),
        "shape": gr.new_vertex_property("int32_t"),
        "size": gr.new_vertex_property("int32_t")
    }
    edges_attr = gr.new_edge_property('double')

    max_weight = 0
    pos = ""

    for s in df.columns:
        nodes[s] = gr.add_vertex() 
        nodes_attr['text'][nodes[s]] = str(s) if number_subs else s
        nodes_attr['color'][nodes[s]] = [0.0,0.0,0.0,0.0]
        nodes_attr['text_color'][nodes[s]] = [0.0,0.0,0.0,1.0]
        nodes_attr['shape'][nodes[s]] = 0
        nodes_attr['size'][nodes[s]] = 10
        
    if 'other' in nodes.keys(): nodes_attr['text'][nodes['other']] = 'X'
    
    if self_loops:
        for s in df.columns:
            for ss in df.index:
                n1 = s 
                n2 = ss 
                
                if n1=='other' and n2=='other' and no_loop_other: continue
                if n1!=n2 and not inc_non_loops: continue
                max_weight = max(max_weight, df.loc[ss][s])

        for s in df.columns:
            for ss in df.index:
                n1 = s 
                n2 = ss 
                
                if n1=='other' and n2=='other' and no_loop_other: continue
                if n1!=n2 and not inc_non_loops: continue

                e = gr.add_edge(nodes[s], nodes[ss]) 
                edges_attr[e] = df.loc[ss][s]/max_weight*10

        if sub!='': 
            pos = sfdp_layout(gr) if not inc_non_loops else radial_tree_layout(gr, nodes[sub])
        else:
            pos = sfdp_layout(gr)
    else:
        for s in df.columns:
            for ss in df.index:
                n1 = s 
                n2 = ss 
                if n1==n2: continue
                max_weight = max(max_weight, df.loc[ss][s])

        for s in df.columns:
            for ss in df.index:
                n1 = s 
                n2 = ss 

                if n1==n2: continue

                e = gr.add_edge(nodes[s], nodes[ss]) 
                edges_attr[e] = df.loc[ss][s]/max_weight*10

        if sub!='': 
            pos = radial_tree_layout(gr, nodes[sub])  
        else:
            pos = sfdp_layout(gr)
        
    o_size = (600,600) if number_subs else (1000,1000)
    draw_graph(gr, pos, nodes_attr, edges_attr, save_graph, o_size)
    
    if number_subs:
        for i, s in enumerate(df.columns):
            print(i, s)


def fetch_top_users_from_file(subreddit: str) -> list:
    json_top_users = json.load(open('scalp/cache/top_users.json', 'r'))
    set_top_users = set() 
    try:
        for d_type in json_top_users[subreddit].keys():
            for date in json_top_users[subreddit][d_type].keys():
                fetchable_top = [a[0] for a in json_top_users[subreddit][d_type][date]['fetchable']]
                set_top_users = set_top_users | set(fetchable_top)
    except Exception as e:
        pass 

    return list(set_top_users)

def fetch_rand_users_from_file(subreddit: str) -> list:
    json_users = json.load(open('scalp/cache/rand_users.json', 'r')) 
    return json_users[subreddit]