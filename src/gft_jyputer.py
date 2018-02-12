from lib.gftTools import gftIO
import pickle

gscall = gftIO.GSCall(server_url='http://172.16.103.106:9080', user_name='wangmeng', pwd='gft')
gid_list = [
            # "CEEC2E875F65937FE4805FDB3328319E",
            # "2F1E1F9031EFA9D32AECB1754A67BD31",
            # "1A779B21C3C81C994DF873B89E22D63C",
            # "38418E70F0EC35CA0BAE433B979948AB",
            # "8517FBA5D741788ABD7E8CD6218A39B4",
            # "0A043423F10EA7CB386C647D96E37467",
            # "A53ED52AFCA5289A8A8666DAC7180A74",
            # "AC26AB3E093237BB5A71FFBDCC3DDF53",
            # "D79CF382423AD43D099E9C92CB526EC2",
            # "02FA150D569F68607160E6F30BDCD96A",
            # "27CA206C5B380465778E4371F02C90E6",
            # "062F39BA60CCB5696254152B22CFAA34",
            # "304BF125406FB2498FFDA45977A91359",
            # "D219ED2B6B386CCACF64316D4452AE15",
            # "E5A012AD99991AD78E09CB4A1FEC2355"
            # "27CA206C5B380465778E4371F02C90E6",
            # "0A043423F10EA7CB386C647D96E37467",
            # "85A86FEF984B28E7A8B6ADF150EBD568",
            "062F39BA60CCB5696254152B22CFAA34",
            "A53ED52AFCA5289A8A8666DAC7180A74",
            "2F1E1F9031EFA9D32AECB1754A67BD31"
            ]
for gid in gid_list:
    graph = gscall.get_graph_from_neo4j(gid)
    # print(graph)
    bin_data_graph = graph.SerializeToString()
    gftIO.zdump(bin_data_graph, '/home/gft/work/gid_' + gid + '.pkl')
