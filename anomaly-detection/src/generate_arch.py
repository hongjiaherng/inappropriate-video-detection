from torchview import draw_graph

from pengwu_net.model import PengWuNet

model = PengWuNet(feature_dim=2048)
model_graph = draw_graph(model, input_size=(5, 100, 2048), expand_nested=True)
model_graph.visual_graph.render("pengwu_net_1.gv", format="svg")
