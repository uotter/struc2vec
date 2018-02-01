import zlib

import lib.gftTools.skill_pb2 as skill
import os as os
import pandas as pd
import gensim as gensim
import networkx as nx
import pickle as pickle


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


def readGraph_from_pickle(filename):
    with open(filename, "rb") as fpz:
        value = fpz.read()
        try:
            ret = pickle.loads(zlib.decompress(value), encoding="latin1")
        except:
            ret = pickle.loads(value)
        ret_graph = skill.RespRunNodeAction()
        ret_graph.ParseFromString(ret)
    return ret_graph


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


def pb2list_networknx(pb_graph_list, output_name):
    edges = []
    nodes = []
    G = nx.Graph()
    for pb_graph in pb_graph_list:
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


def pb2list_edgelist(pb_graph_list, output_name):
    start_node_list = []
    end_node_list = []
    for pb_graph in pb_graph_list:
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


def get_filename_list(pos_file_dir):
    pos_dir_file_list = os.listdir(pos_file_dir)  # 列出文件夹下所有的目录与文件
    pos_file_name_list = []
    print("========Load Files==========")
    print("Files:")
    for i in range(0, len(pos_dir_file_list)):
        path = os.path.join(pos_file_dir, pos_dir_file_list[i])
        # path = unicode(path , "GB2312")
        print(path)
        if os.path.isfile(path):
            pos_file_name_list.append(path)
        else:
            print("File exception:" + path)
    return pos_file_name_list


def get_most_similar(emb_model_dic, nx_g, node_id):
    print("========Get Similar Nodes==========")
    pd.set_option('display.width', 1000)
    df_total = pd.DataFrame()
    columns_list = []
    for key, value in emb_model_dic.items():
        model_name = key
        model_obj = value
        new_model = gensim.models.KeyedVectors.load_word2vec_format(model_obj, binary=False)
        most_similar_list = new_model.most_similar([str(node_id)])
        columns_list.append(key)
        columns_list.append("similarity")
        similar_nodeid_list = []
        similar_nodename_list = []
        similarity_list = []
        for similar_node in most_similar_list:
            similar_node_id = similar_node[0]
            similarity = similar_node[1]
            similar_node_name = nx_g.node[similar_node_id]["name"]
            similar_nodeid_list.append(similar_node_id)
            similar_nodename_list.append(similar_node_name)
            similarity_list.append(similarity)
        df_total.insert(len(df_total.columns), model_name + "_id", pd.Series(similar_nodeid_list))
        df_total.insert(len(df_total.columns), model_name + "_name", pd.Series(similar_nodename_list))
        df_total.insert(len(df_total.columns), model_name + "_similarity", pd.Series(similarity_list))
    print("The nodes most similar to " + node_id + "(" + nx_g.node[node_id]["name"] + ") are:")
    print(df_total)
    # for similar_node in most_similar_list:
    #     similar_node_id = similar_node[0]
    #     similarity = similar_node[1]
    #     similar_node_name = nx_g.node[similar_node_id]["name"]
    #     print(similar_node_id + "(" + similar_node_name + ")    " + str(similarity))


if __name__ == "__main__":
    project_root_path = os.path.abspath('..')
    pickle_file_dir = project_root_path + r"\data\pickle"
    pb2_file_dir = project_root_path + r"\data\pb2"
    edgelist_file_dir = project_root_path + r"\graph"
    gexf_file_dir = project_root_path + r"\data\gexf"
    pb_pickle_filename_list = get_filename_list(pickle_file_dir)
    gexf_total_name = gexf_file_dir + r"\gft_total.gexf"
    edgelist_total_name = gexf_file_dir + r"\gft_total.edgelist"

    pb_name = project_root_path + r"\data\pb2\dump_graph.protobuf.bin"
    edgelist_name = project_root_path + r"\graph\dump_graph2.edgelist"
    struc2vec_model_name = project_root_path + r"\emb\gft_total_struc2vec.emb"
    node2vec_model_name = project_root_path + r"\emb\gft_total_node2vec.emb"
    gexf_name = project_root_path + r"\data\gexf\dump_graph2.gexf"

    emb_model_dic = {"struc2vec": struc2vec_model_name, "node2vec": node2vec_model_name}

    # pb_graph_list = []
    # for pb_pickle_filename in pb_pickle_filename_list:
    #     g = readGraph_from_pickle(pb_pickle_filename)
    #     pb_graph_list.append(g)
    # g = readGraph(pb_name)
    # pb_graph_list.append(g)
    # pb2list_edgelist(pb_graph_list, edgelist_total_name)
    # pb2list_networknx(pb_graph_list, gexf_total_name)
    nx_g = nx.read_gexf(gexf_total_name)
    get_most_similar(emb_model_dic, nx_g, '9412258')
