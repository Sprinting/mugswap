# Split export: tpsmm_src.onnx (once per selfie) + tpsmm_frame.onnx (per frame).
import sys, os, runpy, torch, numpy as np
import torch.nn.functional as F
out_dir = sys.argv[4]
g = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "export.py"), run_name="lib")
inp, dmn, Gen, dm, mcg = g["inp"], g["dmn"], g["Gen"], g["dm"], g["make_coordinate_grid"]

class Src(torch.nn.Module):
    def __init__(s): super().__init__(); s.inp = inp; s.dmn = dmn
    def forward(s, source):
        e = [s.inp.first(source)]
        for b in s.inp.down_blocks: e.append(b(e[-1]))
        return (s.dmn.down(source), *e)

class Frame(torch.nn.Module):
    def __init__(s): super().__init__(); s.inp = inp; s.dmn = dmn
    def forward(s, source, src_small, enc0, enc1, enc2, enc3, kp_driving, kp_source, tps):
        d, n = s.dmn, s.inp
        bs, _, h, w = src_small.shape
        hm = dm.kp2gaussian(kp_driving, (h, w), d.kp_variance) - dm.kp2gaussian(kp_source, (h, w), d.kp_variance)
        hm = torch.cat([torch.zeros(bs, 1, h, w), hm], 1)
        kp_1 = kp_driving.view(bs, -1, 5, 2)
        grid = mcg((h, w), type="torch.FloatTensor").view(1, h * w, 2)
        theta = tps[:, :, 5:, :].permute(0, 1, 3, 2)
        tr = torch.matmul(theta[:, :, :, :2], grid.permute(0, 2, 1)) + theta[:, :, :, 2:]
        dist = ((grid.view(1, 1, 1, -1, 2) - kp_1.view(bs, 10, 5, 1, 2)) ** 2).sum(-1)
        dist = dist * torch.log(dist + 1e-9)
        tr = tr.permute(0, 1, 3, 2) + torch.matmul(dist.permute(0, 1, 3, 2), tps[:, :, :5, :])
        ident = mcg((h, w), type="torch.FloatTensor").view(1, 1, h, w, 2).repeat(bs, 1, 1, 1, 1)
        trans = torch.cat([ident, tr.view(bs, 10, h, w, 2)], 1)
        rep = src_small.unsqueeze(1).repeat(1, 11, 1, 1, 1).view(bs * 11, -1, h, w)
        deformed = F.grid_sample(rep, trans.view(bs * 11, h, w, 2), align_corners=True).view(bs, -1, h, w)
        pred = d.hourglass(torch.cat([hm, deformed], 1), mode=1)
        cmap = F.softmax(d.maps(pred[-1]), dim=1).unsqueeze(2)
        deformation = (trans.permute(0, 1, 4, 2, 3) * cmap).sum(1).permute(0, 2, 3, 1)
        occ = [torch.sigmoid(d.occlusion[i](pred[d.up_nums - d.occlusion_num + i])) for i in range(d.occlusion_num - d.up_nums)]
        p = pred[-1]
        for i in range(d.up_nums):
            p = d.up[i](p); occ.append(torch.sigmoid(d.occlusion[i + d.occlusion_num - d.up_nums](p)))
        # InpaintingNetwork.forward with the source encoder maps precomputed
        enc = [enc0, enc1, enc2, enc3]
        out = n.occlude_input(n.deform_input(enc3, deformation), occ[0])
        for i in range(n.num_down_blocks):
            out = n.resblock[2 * i + 1](n.resblock[2 * i](out))
            out = n.up_blocks[i](out)
            ei = n.occlude_input(n.deform_input(enc[-(i + 2)], deformation), occ[i + 1])
            if i == n.num_down_blocks - 1: break
            out = torch.cat([out, ei], 1)
        ds = n.deform_input(source, deformation)
        out = out * (1 - occ[-1]) + ei
        out = torch.sigmoid(n.final(out))
        return out * (1 - occ[-1]) + ds * occ[-1]

r = np.load(out_dir + "/ref.npz")
T = lambda k: torch.from_numpy(r[k])
with torch.no_grad():
    sv = Src()(T("src"))
    o = Frame()(T("src"), *sv, T("kd"), T("ks"), T("tps"))
    print("split vs original max abs diff", (o - T("out")).abs().max().item())
    kw = dict(opset_version=17, dynamo=False, do_constant_folding=True)
    torch.onnx.export(Src(), (T("src"),), out_dir + "/tpsmm_src.onnx", input_names=["source"],
                      output_names=["src_small", "enc0", "enc1", "enc2", "enc3"], **kw)
    torch.onnx.export(Frame(), (T("src"), *sv, T("kd"), T("ks"), T("tps")), out_dir + "/tpsmm_frame.onnx",
                      input_names=["source", "src_small", "enc0", "enc1", "enc2", "enc3", "kp_driving", "kp_source", "tps"],
                      output_names=["out"], **kw)
