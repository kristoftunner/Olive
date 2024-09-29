import onnxruntime as ort
from time import time
import os
import numpy as np

if __name__ == '__main__':
    olive_path = 'C:/Users/kristof/dev/repos/Olive/examples/stable_diffusion/models/optimized-dml/stable-diffusion-v1-5/stable-diffusion-v1-5'
    text_encoder_model_path = os.path.join(
        olive_path, "text_encoder/model.onnx")
    vae_decoder_path = os.path.join(
        olive_path, "vae_decoder/model.onnx")
    unet_path = os.path.join(
        olive_path, "unet/model.onnx")
    controlnet_path = os.path.join(
        olive_path, "controlnet/model.onnx")
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    session_options.enable_mem_pattern = False

    text_encoder_session = ort.InferenceSession(
        text_encoder_model_path,
        providers=["DmlExecutionProvider"],
        sess_options=session_options)
    vae_decoder_session = ort.InferenceSession(
        vae_decoder_path,
        providers=["DmlExecutionProvider"],
        sess_options=session_options)
    unet_session = ort.InferenceSession(
        unet_path,
        providers=["DmlExecutionProvider"],
        sess_options=session_options)
    controlnet_session = ort.InferenceSession(
        controlnet_path,
        providers=["DmlExecutionProvider"],
        sess_options=session_options)
    
    unet_input_tensors = {
        "sample" : np.random.rand(1,4,64,64).astype(np.float16),
        "timestep" : np.random.rand(1).astype(np.float16),
        "encoder_hidden_states" : np.random.rand(1,77,768).astype(np.float16),
        "mid_block_0"  : np.random.rand(1,320,64,64).astype(np.float16),
        "mid_block_1"  : np.random.rand(1,320,64,64).astype(np.float16),
        "mid_block_2"  : np.random.rand(1,320,64,64).astype(np.float16),
        "mid_block_3"  : np.random.rand(1,320,32,32).astype(np.float16),
        "mid_block_4"  : np.random.rand(1,640,32, 32).astype(np.float16),
        "mid_block_5"  : np.random.rand(1,640,32, 32).astype(np.float16),
        "mid_block_6"  : np.random.rand(1,640,16, 16).astype(np.float16),
        "mid_block_7"  : np.random.rand(1,1280,16,16).astype(np.float16),
        "mid_block_8"  : np.random.rand(1,1280,16,16).astype(np.float16),
        "mid_block_9"  : np.random.rand(1,1280,8,8).astype(np.float16),
        "mid_block_10" : np.random.rand(1,1280,8,8).astype(np.float16),
        "mid_block_11" : np.random.rand(1,1280,8,8).astype(np.float16),
        "down_block": np.random.rand(1280, 8, 8).astype(np.float16),
    }
    unet_avg_time = 0
    for i in range(100):
        start = time()
        unet_output = unet_session.run([], unet_input_tensors)
        stop = time()
        unet_avg_time += stop - start
    unet_avg_time /= 100
    print(f"Average time: {unet_avg_time}")

    controlnet_input_tensors = {
        "sample" : np.random.rand(1,4,64,64).astype(np.float16),
        "timestep" : np.random.rand(1).astype(np.float16),
        "encoder_hidden_states" : np.random.rand(1,77,768).astype(np.float16),
        "controlnet_cond" : np.random.rand(1,3,512,512).astype(np.float16),
        "conditioning_scale" : np.random.rand(1,1).astype(np.float16),
    }

    controlnet_avg_time = 0
    for i in range(100):
        start = time()
        controlnet_output = controlnet_session.run([], controlnet_input_tensors)
        stop = time()
        controlnet_avg_time += stop - start
    controlnet_avg_time /= 100
    print(f"Average time: {controlnet_avg_time}")

    text_encoder_input_tensors = {
        "input_ids" : np.random.randint(0, 50256, (1, 77)).astype(np.int32),
    }

    text_encoder_avg_time = 0
    for i in range(100):
        start = time()
        text_encoder_output = text_encoder_session.run([], text_encoder_input_tensors)
        stop = time()
        text_encoder_avg_time += stop - start
    text_encoder_avg_time /= 100
    print(f"Average time: {text_encoder_avg_time}")


    print("Models loaded into ORT sessions.")
    