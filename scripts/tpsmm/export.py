# Export TPSMM (vox 256) to ONNX: kp_detector.onnx and generator.onnx.
# The TPS solve (8x8 inverse per TPS) is done outside the graph (JS) and fed in as `tps`.
import sys, torch, yaml, numpy as np
sys.path.insert(0, sys.argv[1])  # TPSMM repo
from modules.inpainting_network import InpaintingNetwork
from modules.keypoint_detector import KPDetector
from modules.dense_motion import DenseMotionNetwork
import modules.dense_motion as dm
from modules.util import make_coordinate_grid
import torch.nn.functional as F

cfg = yaml.full_load(open(sys.argv[2]))
ck = torch.load(sys.argv[3], map_location="cpu", weights_only=False)
out_dir = sys.argv[4]
mp = cfg["model_params"]
inp = InpaintingNetwork(**mp["generator_params"], **mp["common_params"]).eval()
kpd = KPDetector(**mp["common_params"]).eval()
dmn = DenseMotionNetwork(**mp["common_params"], **mp["dense_motion_params"]).eval()
inp.load_state_dict(ck["inpainting_network"]); kpd.load_state_dict(ck["kp_detector"]); dmn.load_state_dict(ck["dense_motion_network"])

def tps_param(kp_1, kp_2):
    """Reference TPS solve (torch.inverse), same as modules.util.TPS(mode='kp')."""
    bs = kp_1.shape[0]; n = kp_1.shape[2]
    K = torch.norm(kp_1[:, :, :, None] - kp_1[:, :, None, :], dim=4, p=2) ** 2
    K = K * torch.log(K + 1e-9)
    one1 = torch.ones(bs, kp_1.shape[1], n, 1)
    kp_1p = torch.cat([kp_1, one1], 3)
    P = torch.cat([kp_1p, torch.zeros(bs, kp_1.shape[1], 3, 3)], 2)
    L = torch.cat([torch.cat([K, kp_1p.permute(0, 1, 3, 2)], 2), P], 3)
    Y = torch.cat([kp_2, torch.zeros(bs, kp_1.shape[1], 3, 2)], 2)
    L = L + torch.eye(L.shape[2]).expand(L.shape) * 0.01
    return torch.matmul(torch.inverse(L), Y)  # [bs, 10, 8, 2]

class Gen(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.inp = inp; s.dmn = dmn
    def forward(s, source, kp_driving, kp_source, tps):
        d = s.dmn
        src = d.down(source)
        bs, _, h, w = src.shape  # 64x64
        # heatmaps
        hm = (dm.kp2gaussian(kp_driving, (h, w), d.kp_variance) - dm.kp2gaussian(kp_source, (h, w), d.kp_variance))
        hm = torch.cat([torch.zeros(bs, 1, h, w), hm], 1)
        # TPS warps of the identity grid
        kp_1 = kp_driving.view(bs, -1, 5, 2)
        grid = make_coordinate_grid((h, w), type="torch.FloatTensor").view(1, h * w, 2)
        theta = tps[:, :, 5:, :].permute(0, 1, 3, 2)          # [bs,10,2,3]
        cparams = tps[:, :, :5, :]                             # [bs,10,5,2]
        tr = torch.matmul(theta[:, :, :, :2], grid.permute(0, 2, 1)) + theta[:, :, :, 2:]
        dist = ((grid.view(1, 1, 1, -1, 2) - kp_1.view(bs, 10, 5, 1, 2)) ** 2).sum(-1)
        dist = dist * torch.log(dist + 1e-9)
        tr = tr.permute(0, 1, 3, 2) + torch.matmul(dist.permute(0, 1, 3, 2), cparams)
        d2s = tr.view(bs, 10, h, w, 2)
        ident = make_coordinate_grid((h, w), type="torch.FloatTensor").view(1, 1, h, w, 2).repeat(bs, 1, 1, 1, 1)
        trans = torch.cat([ident, d2s], 1)                     # [bs,11,h,w,2]
        rep = src.unsqueeze(1).repeat(1, 11, 1, 1, 1).view(bs * 11, -1, h, w)
        deformed = F.grid_sample(rep, trans.view(bs * 11, h, w, 2), align_corners=True).view(bs, -1, h, w)
        pred = d.hourglass(torch.cat([hm, deformed], 1), mode=1)
        cmap = F.softmax(d.maps(pred[-1]), dim=1).unsqueeze(2)
        deformation = (trans.permute(0, 1, 4, 2, 3) * cmap).sum(1).permute(0, 2, 3, 1)
        occ = []
        for i in range(d.occlusion_num - d.up_nums):
            occ.append(torch.sigmoid(d.occlusion[i](pred[d.up_nums - d.occlusion_num + i])))
        p = pred[-1]
        for i in range(d.up_nums):
            p = d.up[i](p)
            occ.append(torch.sigmoid(d.occlusion[i + d.occlusion_num - d.up_nums](p)))
        out = s.inp(source, {"contribution_maps": None, "deformed_source": None, "occlusion_map": occ, "deformation": deformation})
        return out["prediction"]

class KP(torch.nn.Module):
    def __init__(s): super().__init__(); s.k = kpd
    def forward(s, image): return s.k(image)["fg_kp"]

if __name__ == "__main__":
    torch.manual_seed(0)
    src = torch.rand(1, 3, 256, 256); drv = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        ks, kd = kpd(src)["fg_kp"], kpd(drv)["fg_kp"]
        tps = tps_param(kd.view(1, -1, 5, 2), ks.view(1, -1, 5, 2))
        ref = inp(src, dmn(src, {"fg_kp": kd}, {"fg_kp": ks}, None, False))["prediction"]
        mine = Gen()(src, kd, ks, tps)
        print("wrapper vs original max abs diff", (ref - mine).abs().max().item())
        np.savez(out_dir + "/ref.npz", src=src.numpy(), drv=drv.numpy(), ks=ks.numpy(), kd=kd.numpy(), tps=tps.numpy(), out=ref.numpy())
        kw = dict(opset_version=17, dynamo=False, do_constant_folding=True)
        torch.onnx.export(KP(), (src,), out_dir + "/tpsmm_kp.onnx", input_names=["image"], output_names=["kp"], **kw)
        torch.onnx.export(Gen(), (src, kd, ks, tps), out_dir + "/tpsmm_gen.onnx",
                          input_names=["source", "kp_driving", "kp_source", "tps"], output_names=["out"], **kw)
    print("ok")
