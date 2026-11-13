# TPSMM ONNX export (spike)

Rebuilds `public/models/tpsmm/` from the Thin-Plate Spline Motion Model vox-256 checkpoint.

```sh
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxruntime pyyaml scipy
git clone https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model tpsmm
HF=https://huggingface.co/MAGIG/thin-plate-spline-motion-model/resolve/d1a59ac77161d35044b0e06c879266aafd35f51b/256
curl -L -o vox.pth.tar $HF/vox.pth.tar && curl -L -o vox-256.yaml $HF/vox-256.yaml
mkdir -p out
python export.py       tpsmm vox-256.yaml vox.pth.tar out   # reference outputs + one-piece model (sanity check)
python export_split.py tpsmm vox-256.yaml vox.pth.tar out   # fp32: tpsmm_kp, tpsmm_src, tpsmm_frame
python export_fp16.py  tpsmm vox-256.yaml vox.pth.tar out   # fp16 versions (fp32 inputs/outputs)
python check.py out                                         # compares both against PyTorch
cd out && split -b 50000000 -a 2 tpsmm_frame.onnx tpsmm_frame.onnx.part- \
       && split -b 50000000 -a 2 tpsmm_frame_fp16.onnx tpsmm_frame_fp16.onnx.part-
```

Models: `tpsmm_kp` (crop → 50 keypoints), `tpsmm_src` (selfie → feature maps, once),
`tpsmm_frame` (features + keypoints + spline params → frame). The 10 thin-plate spline
solves (8×8 each) happen in JS (`src/spike/tpsmm.ts`) because ONNX's matrix inverse
doesn't run on WebGPU. The checkpoint and code are MIT licensed.
