"""Label conventions shared by the whole code base."""

# AI4Arctic ready-to-train "SOD" (stage of development) classes.
SOD_CLASSES = ["open water", "new ice", "young ice", "thin FYI", "thick FYI", "old ice"]
N_SOD = len(SOD_CLASSES)

# Pixels with this label are land, no-data or ambiguous chart polygons: they are
# excluded from both the loss and every metric.
IGNORE = 255

# Skadi product classes (Section 2.2 of the mission document). The network is
# trained on the six SOD classes and the three-class map is obtained by summing
# the class probabilities of each group.
SKADI_CLASSES = ["OW", "FYI", "MYI"]
SOD_TO_SKADI = [0, 1, 1, 1, 1, 2]
MYI_INDEX = 5  # "old ice" in the SOD convention

# Reference numbers reported by TinyIceNet (arXiv:2603.03075, Tables 1 and 4).
TINYICENET_PAPER = {
    "f1_fp32": 75.168,
    "f1_qat_int8": 75.216,
    "params": 146_600,
    "gmacs": 2.97,
}
