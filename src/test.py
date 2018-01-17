import lib.gftTools.skill_pb2 as skill
import os as os
import pandas as pd
import gensim as gensim
import networkx as nx


def readGraph(file_name):
    with open(file_name, 'rb', -1) as fps:
        bin_datas = fps.read()
        ret_graph = skill.RespRunNodeAction()
        ret_graph.ParseFromString(bin_datas)
        return ret_graph


def pb2edgelist(pb_graph, output_name):
    start_node_list = []
    end_node_list = []
    graph_num = len(pb_graph.graphs._values)
    for g_index in range(graph_num):
        graph = pb_graph.graphs[g_index].graph
        edge_num = len(graph.edges._values)
        for e_index in range(edge_num):
            edge = graph.edges[e_index]
            start_node_list.append(edge.sn_id)
            end_node_list.append(edge.en_id)
    ret_df = pd.DataFrame([start_node_list, end_node_list]).T
    ret_df.to_csv(output_name, sep=" ", header=False, index=False)


def pb2networknx(pb_graph, output_name):
    edges = []
    nodes = []
    G = nx.Graph()
    graph_num = len(pb_graph.graphs._values)
    for g_index in range(graph_num):
        graph = pb_graph.graphs[g_index].graph
        edge_num = len(graph.edges._values)
        node_num = len(graph.nodes._values)
        for e_index in range(edge_num):
            edge = graph.edges[e_index]
            edges.append([edge.sn_id, edge.en_id])
        for n_index in range(node_num):
            node_entity = graph.nodes[n_index]
            n_id = node_entity.nid
            n_name = node_entity.node_prop.props.entries[-2].value
            nodes.append((n_id, {"name": n_name}))
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    nx.write_gexf(G, output_name)
    # return G


def get_most_similar(model_name, nx_g, node_id):
    new_model = gensim.models.KeyedVectors.load_word2vec_format(model_name, binary=False)
    most_similar_list = new_model.most_similar([str(node_id)])
    print(node_id + "(" + nx_g.node[node_id]["name"] + ") results:")
    for similar_node in most_similar_list:
        similar_node_id = similar_node[0]
        similarity = similar_node[1]
        similar_node_name = nx_g.node[similar_node_id]["name"]
        print(similar_node_id + "(" + similar_node_name + ")    " + str(similarity))


if __name__ == "__main__":
    project_root_path = os.path.abspath('..')
    pb_name = project_root_path + r"\data\dump_graph.protobuf.bin"
    degelist_name = project_root_path + r"\graph\dump_graph.edgelist"
    model_name = project_root_path + r"\emb\dump_graph.emb"
    gexf_name = project_root_path + r"\data\dump_graph.gexf"
    # g = readGraph(pb_name)
    nx_g = nx.read_gexf(gexf_name)
    # pb2edgelist(g, degelist_name)
    get_most_similar(model_name, nx_g, '9240087')
    # pb2networknx(g, gexf_name)
