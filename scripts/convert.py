"""Convert ONNX to .pt. Uses onnx2torch when possible; falls back to ONNX Runtime wrapper for models with Loop (e.g. NMS)."""
import torch
import onnx

from onnx_wrapper import OnnxRuntimeWrapper

onnx_model_path = '/workspace/ros2_workspace/src/rosetta/scripts/models/face.onnx'
pytorch_model_path = '/workspace/ros2_workspace/src/rosetta/scripts/models/face.pt'


def main():
    try:
        from onnx2torch import convert
        from onnxsim import simplify
        model = onnx.load(onnx_model_path)
        model_simp, check = simplify(model)
        if not check:
            print("Warning: onnxsim could not verify simplified model")
        onnx_simp_path = onnx_model_path.replace('.onnx', '_simp.onnx')
        onnx.save(model_simp, onnx_simp_path)
        pytorch_model = convert(onnx_simp_path)
    except NotImplementedError as e:
        if 'Loop' in str(e):
            print(f"onnx2torch doesn't support Loop op. Using ONNX Runtime wrapper.")
            pytorch_model = OnnxRuntimeWrapper(onnx_model_path)
        else:
            raise
    torch.save(pytorch_model, pytorch_model_path)
    print(f"Saved: {pytorch_model_path}")


if __name__ == '__main__':
    main()
