import sys, torch
sys.path.insert(0, "D:/USV-landing-prediction/experiments/phylstm")
sys.argv = ["x"]
import phylstm_kaggle as pk
def draw(build2, dev):
    torch.manual_seed(0)
    n1 = pk.Stack(4, 9).to(dev)
    if build2: pk.Stack(13, 3).to(dev)
    o = torch.randint(300, 1000, (5000,), device=dev); p = torch.randperm(5000 * 4, device=dev)
    o2 = torch.randint(300, 1000, (5000,), device=dev)
    return n1.d2.weight.detach().cpu(), o.cpu(), p.cpu(), o2.cpu()
print(torch.__version__)
for dev in ["cuda", "cpu"]:
    a, b = draw(False, dev), draw(True, dev)
    print(dev, "init equal", torch.equal(a[0], b[0]), "randint equal", torch.equal(a[1], b[1]), "randperm equal", torch.equal(a[2], b[2]), "2nd randint equal", torch.equal(a[3], b[3]))
