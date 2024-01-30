from torchview import draw_graph

from pengwu_net.model import PengWuNet
from svm_baseline.model import BaselineNet
from sultani_net.model import SultaniNet

hlnet = PengWuNet(feature_dim=2048)
hlnet_graph = draw_graph(
    hlnet,
    input_size=(32, 100, 2048),
    expand_nested=True,
)
hlnet_graph.visual_graph.render("hlnet_1.gv", format="png")


baseline = BaselineNet(feature_dim=2048)
baseline_graph = draw_graph(
    baseline,
    input_size=(32, 2048),
    expand_nested=True,
)
baseline_graph.visual_graph.render("baseline_1.gv", format="png")


sultani = SultaniNet(feature_dim=2048)
sultani_graph = draw_graph(
    sultani,
    input_size=(32, 2048),
    expand_nested=True,
)
sultani_graph.visual_graph.render("sultani_1.gv", format="png")
