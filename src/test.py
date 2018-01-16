import lib.gftTools.skill_pb2 as skill
import os as os
import pandas as pd


def readGraph(file_name):
    with open(file_name, 'rb', -1) as fps:
        bin_datas = fps.read()
        ret_graph = skill.RespRunNodeAction()
        ret_graph.ParseFromString(bin_datas)
        return ret_graph


def pb2edgelist(pb_graph,output_name):
    start_node_list = []
    end_node_list = []
    graph_num = len(g.graphs._values)
    for g_index in range(graph_num):
        graph = g.graphs[g_index].graph
        edge_num = len(graph.edges._values)
        for e_index in range(edge_num):
            edge = graph.edges[e_index]
            start_node_list.append(edge.sn_id)
            end_node_list.append(edge.en_id)
    ret_df = pd.DataFrame([start_node_list,end_node_list]).T
    ret_df.to_csv(output_name,sep=" ",header=False,index=False)



if __name__ == "__main__":
    project_root_path = os.path.abspath('..')
    pb_name = project_root_path + r"\data\dump_graph.protobuf.bin"
    degelist_name = project_root_path + r"\graph\dump_graph.edgelist"
    g = readGraph(pb_name)
    pb2edgelist(g, degelist_name)

