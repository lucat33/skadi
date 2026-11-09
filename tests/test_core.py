"""Unit tests that need neither the dataset nor a GPU:  python -m pytest -q"""
import numpy as np
import pytest
import torch

from skadi.constants import IGNORE, N_SOD
from skadi.data import Augment
from skadi.infer import confidence_from_probs, tiled_predict, vectorize
from skadi.losses import CEDiceLoss, class_weights
from skadi.metrics import collapse, confusion, scores
from skadi.models import SkadiUNet, TinyIceNet, count_macs, count_params
from skadi.preprocess import resize_area, resize_nearest


def test_tinyicenet_matches_paper_size():
    net = TinyIceNet(in_ch=2, n_classes=7)  # the paper uses a 7-way head
    assert count_params(net) == 146_599  # "146.6 k parameters"
    assert 2.8e9 < count_macs(net) < 3.0e9  # "2.97 GMac"


def test_skadi_unet_shape_and_cost():
    net = SkadiUNet(in_ch=2, pretrained=False).eval()
    out = net(torch.zeros(1, 2, 256, 320))
    assert out.shape == (1, N_SOD, 256, 320)
    assert count_macs(net) < 0.5 * count_macs(TinyIceNet())
    assert count_params(net) < 5e6  # budget of the Skadi document


def test_metrics_agree_with_sklearn():
    f1_score = pytest.importorskip("sklearn.metrics").f1_score
    rng = np.random.RandomState(0)
    target = rng.choice([0, 1, 2, 3, 4, 5, IGNORE], size=20_000, p=[.5, .05, .05, .1, .15, .05, .1])
    pred = np.where(rng.rand(target.size) < 0.7, np.minimum(target, 5), rng.randint(0, 6, target.size))
    cm = confusion(torch.from_numpy(pred), torch.from_numpy(target))
    s = scores(cm)
    keep = target != IGNORE
    assert s["f1_weighted"] == pytest.approx(100 * f1_score(target[keep], pred[keep], average="weighted"), abs=1e-6)
    assert s["f1_macro"] == pytest.approx(100 * f1_score(target[keep], pred[keep], average="macro"), abs=1e-6)
    assert cm.sum() == keep.sum()


def test_three_class_collapse():
    cm = torch.arange(36).view(6, 6)
    c3 = collapse(cm)
    assert c3.sum() == cm.sum()
    assert c3[0, 0] == cm[0, 0] and c3[2, 2] == cm[5, 5]
    assert c3[1, 1] == cm[1:5, 1:5].sum()


def test_loss_behaviour():
    target = torch.randint(0, N_SOD, (2, 32, 32))
    target[:, :4] = IGNORE
    weights = class_weights(torch.tensor([1e6, 1e3, 1e4, 1e5, 1e5, 1e4]))
    assert weights.mean() == pytest.approx(1.0)
    assert weights[1] > weights[0]  # rare classes weigh more
    loss = CEDiceLoss(weights)
    perfect = 20 * torch.nn.functional.one_hot(target.clamp(max=5), N_SOD).permute(0, 3, 1, 2).float()
    random = torch.randn(2, N_SOD, 32, 32, requires_grad=True)
    assert loss(perfect, target) < 1e-3
    value = loss(random, target)
    value.backward()
    assert value > 1 and random.grad[:, :, :4].abs().sum() == 0  # ignored pixels get no gradient
    assert loss(random, torch.full_like(target, IGNORE)) == 0


def test_augment_keeps_labels_valid():
    torch.manual_seed(0)
    x = torch.randn(4, 8, 64, 64)
    y = torch.randint(0, N_SOD, (4, 64, 64))
    y[:, :8] = IGNORE
    xa, ya = Augment()(x, y)
    assert xa.shape == x.shape and ya.shape == y.shape and ya.dtype == torch.long
    assert set(ya.unique().tolist()) <= set(range(N_SOD)) | {IGNORE}
    assert torch.isfinite(xa).all()


def test_resize_handles_nan_and_labels():
    img = np.ones((100, 130), np.float32)
    img[:, :65] = np.nan
    out, frac = resize_area(img, 32)
    assert out.shape == (32, 32) and np.isfinite(out).all()
    assert out[:, 20:].min() == pytest.approx(1.0) and frac[:, :10].max() == 0
    lab = np.full((100, 130), 3, np.uint8)
    lab[:50] = IGNORE
    assert set(np.unique(resize_nearest(lab, 32)).tolist()) == {3, IGNORE}


def test_tiled_prediction_and_confidence():
    net = TinyIceNet().eval()
    x = torch.randn(1, 2, 96, 160)
    direct = net(x).softmax(1)
    tiled = tiled_predict(net, x, tile=64, overlap=16)
    assert tiled.shape == direct.shape
    assert torch.allclose(tiled.sum(1), torch.ones(1, 96, 160), atol=1e-4)
    conf = confidence_from_probs(torch.tensor([[1.0, 0, 0], [1 / 3, 1 / 3, 1 / 3]])[:, :, None, None])
    assert conf[0].item() == pytest.approx(1.0, abs=1e-5) and conf[1].item() == pytest.approx(0.0, abs=1e-5)


def test_vectorize_polygons():
    pytest.importorskip("cv2")
    cls = np.zeros((64, 64), np.uint8)
    cls[10:40, 10:40] = 2  # a square of multi-year ice in open water
    conf = np.full((64, 64), 0.9, np.float32)
    product = vectorize(cls, conf, (10.0, 10.0))
    kinds = sorted(f["properties"]["class"] for f in product["features"])
    assert kinds == ["MYI", "OW"]
    myi = next(f for f in product["features"] if f["properties"]["class"] == "MYI")
    xs = [p[0] for p in myi["geometry"]["coordinates"][0]]
    assert min(xs) == 100 and max(xs) == 390  # scaled to scene pixels
    assert not myi["properties"]["low_confidence"]


def test_augment_moves_image_and_labels_together():
    """The geometric warp must keep every label on top of the pixel it describes."""
    torch.manual_seed(1)
    blocks = torch.randint(0, N_SOD, (6, 1, 8, 8)).float()
    y = torch.nn.functional.interpolate(blocks, scale_factor=16, mode="nearest")[:, 0].long()
    x = y.float().unsqueeze(1).repeat(1, 2, 1, 1)  # the "image" is the label map itself
    aug = Augment()
    xa, ya = aug.geometric(x, y)
    inside = ya != IGNORE
    agree = (xa[:, 0].round().long() == ya) & inside
    # disagreement is only possible on the interpolated borders between blocks
    assert inside.float().mean() > 0.3
    assert agree.sum() / inside.sum() > 0.93
