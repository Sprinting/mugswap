import onnxruntime as ort, numpy as np, sys
d = sys.argv[1]; r = np.load(d + "/ref.npz")
for suf in ["", "_fp16"]:
    s = ort.InferenceSession(f"{d}/tpsmm_src{suf}.onnx"); sv = s.run(None, {"source": r["src"]})
    f = ort.InferenceSession(f"{d}/tpsmm_frame{suf}.onnx")
    names = ["src_small","enc0","enc1","enc2","enc3"]
    o = f.run(None, {"source": r["src"], **dict(zip(names, sv)), "kp_driving": r["kd"], "kp_source": r["ks"], "tps": r["tps"]})[0]
    k = ort.InferenceSession(f"{d}/tpsmm_kp{suf}.onnx").run(None, {"image": r["src"]})[0]
    print(suf or "fp32", "out diff max", np.abs(o - r["out"]).max(), "mean", np.abs(o - r["out"]).mean(), "kp diff", np.abs(k - r["ks"]).max())
