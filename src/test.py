import lib.gftTools.skill_pb2 as skill
import os as os


def readGraph(file_name):
    with open(file_name, 'rb', -1) as fps:
        bin_datas = fps.read()
        ret_graph = skill.RespRunNodeAction()
        ret_graph.ParseFromString(bin_datas)
        return ret_graph


if __name__ == "__main__":
    project_root_path = os.path.abspath('..')
    file_name = project_root_path + r"\data\dump_graph.protobuf.bin"
    g = readGraph(file_name)
    print(g)
