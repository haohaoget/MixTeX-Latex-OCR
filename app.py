import streamlit as st
from PIL import Image, ImageGrab
import onnxruntime as ort
import numpy as np
import re
from transformers import AutoTokenizer, AutoImageProcessor
import torch

# --- Core Functions from mixtex_core.py ---

@st.cache_resource
def load_model(model_dir, use_gpu):
    providers = ort.get_available_providers()
    provider = "CPUExecutionProvider"
    if use_gpu and "CUDAExecutionProvider" in providers:
        provider = "CUDAExecutionProvider"
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    feature_extractor = AutoImageProcessor.from_pretrained(model_dir, size=448)
    encoder_sess = ort.InferenceSession(f"{model_dir}/encoder_model.onnx", providers=[provider])
    decoder_sess = ort.InferenceSession(f"{model_dir}/decoder_model_merged.onnx", providers=[provider])
    return (tokenizer, feature_extractor, encoder_sess, decoder_sess), provider

def check_repetition(s, repeats=12):
    for pattern_length in range(1, len(s) // repeats + 1):
        for start in range(len(s) - repeats * pattern_length + 1):
            pattern = s[start : start + pattern_length]
            if s[start : start + repeats * pattern_length] == pattern * repeats:
                return True
    return False

def stream_inference(
    image, model, max_length=512, num_layers=6, hidden_size=768, heads=12, batch_size=1
):
    tokenizer, feature_extractor, enc_session, dec_session = model
    head_size = hidden_size // heads
    inputs = feature_extractor(image, return_tensors="np").pixel_values
    enc_out = enc_session.run(None, {"pixel_values": inputs})[0]
    dec_in = {
        "input_ids": tokenizer("<s>", return_tensors="np").input_ids.astype(np.int64),
        "encoder_hidden_states": enc_out,
        "use_cache_branch": np.array([True], dtype=bool),
        **{
            f"past_key_values.{i}.{t}": np.zeros(
                (batch_size, heads, 0, head_size), dtype=np.float32
            )
            for i in range(num_layers)
            for t in ["key", "value"]
        },
    }
    generated = ""
    for _ in range(max_length):
        outs = dec_session.run(None, dec_in)
        next_id = np.argmax(outs[0][:, -1, :], axis=-1)
        token_text = tokenizer.decode(next_id, skip_special_tokens=True)
        yield token_text
        generated += token_text
        if check_repetition(generated, 21) or next_id == tokenizer.eos_token_id:
            break
        dec_in.update(
            {
                "input_ids": next_id[:, None],
                **{
                    f"past_key_values.{i}.{t}": outs[i * 2 + 1 + j]
                    for i in range(num_layers)
                    for j, t in enumerate(["key", "value"])
                },
            }
        )

# --- Streamlit UI ---

def run_inference(model, img):
    st.image(img, caption="图片预览")
    
    with st.spinner("正在识别..."):
        partial_result = ""
        output_area = st.empty()
        # The feature_extractor (AutoImageProcessor) will handle resizing and padding
        for piece in stream_inference(img, model):
            partial_result += piece
            output_area.code(partial_result, language="latex")

def main():
    st.set_page_config(page_title="MixTeX LaTeX OCR", page_icon="demo/icon.png")
    st.title("MixTeX - 图片转 LaTeX")

    # Sidebar for options
    st.sidebar.header("选项")
    
    # GPU option
    use_gpu_option = False
    if torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers():
        use_gpu_option = st.sidebar.checkbox("使用 GPU 加速", value=True)
    else:
        st.sidebar.warning("未检测到可用的 CUDA GPU。将使用 CPU。")

    model, provider = load_model("onnx", use_gpu_option)
    st.toast(f"正在使用 {provider} 进行推理。")

    # Main content
    st.info("可以通过上传文件、拖拽图片或从剪贴板粘贴来进行识别。")
    
    uploaded_files = st.file_uploader(
        "选择或拖拽图片文件", type=["png", "jpg", "jpeg"], accept_multiple_files=True
    )

    if st.button("从剪贴板粘贴图片"):
        try:
            img = ImageGrab.grabclipboard()
            if img:
                run_inference(model, img)
            else:
                st.warning("剪贴板中没有图片。")
        except Exception as e:
            st.error(f"粘贴失败: {e}")

    if uploaded_files:
        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file).convert("RGB")
            run_inference(model, img)


if __name__ == "__main__":
    main()
