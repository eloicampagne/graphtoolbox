"""Fixed color of each operator shown in the aggregation and regional-mixture figures.

One operator keeps the same color in both figures. The eight categorical slots of the
paper's palette are assigned so that the five experts of the aggregation figure have
distinct hues and that neighboring regions of the map, labeled by their heaviest expert,
do not share a near hue (Normandie borders Bretagne, Ile-de-France and Hauts-de-France).
"""
OPERATOR_COLORS = {
    "GatedGraphConv": "#2a78d6", "GENConv": "#6250d6", "GCNConv": "#1baf7a", "GATv2Conv": "#eda100",
    "RGATConv": "#e87ba4", "PNAConv": "#eb6834", "SSGConv": "#e34948", "FeaStConv": "#008300",
}


def operator_color(name):
    """Color of an operator; fails loudly when a new operator needs a slot."""
    if name not in OPERATOR_COLORS:
        raise KeyError(f"{name} has no color in operator_colors.py")
    return OPERATOR_COLORS[name]
