import gradio as gr
import os
import cv2
from diffsynth import SDVideoPipelineRunner, download_models
import tempfile
import math

def adjust_dimension(dim):
    """调整尺寸为64的倍数"""
    return int(math.ceil(dim / 64)) * 64

def get_video_info(video_path):
    """获取视频的基本信息"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    
    # 调整尺寸为64的倍数
    adjusted_width = adjust_dimension(width)
    adjusted_height = adjust_dimension(height)
    
    return width, height, adjusted_width, adjusted_height, fps, total_frames

def process_video(
    input_video,
    prompt,
    negative_prompt,
    cfg_scale,
    denoising_strength,
    num_inference_steps,
    seed,
    tile_scale,
    lineart_scale
):
    # 创建临时输出目录
    output_dir = tempfile.mkdtemp()
    
    # 获取视频信息
    if input_video is None:
        raise gr.Error("请先上传输入视频")
    
    # 获取视频的真实参数
    try:
        orig_width, orig_height, width, height, fps, total_frames = get_video_info(input_video)
        print(f"原始视频尺寸: {orig_width}x{orig_height}")
        print(f"调整后尺寸: {width}x{height}")
        print(f"FPS: {fps}, 总帧数: {total_frames}")
        
        # 如果尺寸有调整，显示提示信息
        if orig_width != width or orig_height != height:
            gr.Info(f"视频尺寸已从 {orig_width}x{orig_height} 调整为 {width}x{height} 以满足模型要求")
            
    except Exception as e:
        raise gr.Error(f"无法读取视频信息: {str(e)}")
    
    # 确保参数合理
    if fps <= 0:
        fps = 30  # 默认值
    if total_frames <= 0:
        total_frames = 30  # 默认值
    
    # 下载模型（注释掉自动下载，改为按需下载）
    download_models([
        "AingDiffusion_v12",
        "AnimateDiff_v2",
        "ControlNet_v11p_sd15_lineart",
        "ControlNet_v11f1e_sd15_tile",
        "TextualInversion_VeryBadImageNegative_v1.3"
    ])

    config = {
        "models": {
            "model_list": [
                "models/stable_diffusion/aingdiffusion_v12.safetensors",
                "models/AnimateDiff/mm_sd_v15_v2.ckpt",
                "models/ControlNet/control_v11f1e_sd15_tile.pth",
                "models/ControlNet/control_v11p_sd15_lineart.pth"
            ],
            "textual_inversion_folder": "models/textual_inversion",
            "device": "cuda",
            "lora_alphas": [],
            "controlnet_units": [
                {
                    "processor_id": "tile",
                    "model_path": "models/ControlNet/control_v11f1e_sd15_tile.pth",
                    "scale": tile_scale
                },
                {
                    "processor_id": "lineart",
                    "model_path": "models/ControlNet/control_v11p_sd15_lineart.pth",
                    "scale": lineart_scale
                }
            ]
        },
        "data": {
            "input_frames": {
                "video_file": input_video,
                "image_folder": None,
                "height": height,
                "width": width,
                "start_frame_id": 0,
                "end_frame_id": total_frames - 1  # 帧索引从0开始
            },
            "controlnet_frames": [
                {
                    "video_file": input_video,
                    "image_folder": None,
                    "height": height,
                    "width": width,
                    "start_frame_id": 0,
                    "end_frame_id": total_frames - 1
                },
                {
                    "video_file": input_video,
                    "image_folder": None,
                    "height": height,
                    "width": width,
                    "start_frame_id": 0,
                    "end_frame_id": total_frames - 1
                }
            ],
            "output_folder": output_dir,
            "fps": fps
        },
        "pipeline": {
            "seed": seed,
            "pipeline_inputs": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "cfg_scale": cfg_scale,
                "clip_skip": 2,
                "denoising_strength": denoising_strength,
                "num_inference_steps": num_inference_steps,
                "animatediff_batch_size": 16,
                "animatediff_stride": 8,
                "unet_batch_size": 1,
                "controlnet_batch_size": 1,
                "cross_frame_attention": False,
                "input_frames": [],
                "num_frames": total_frames,
                "width": width,
                "height": height,
                "controlnet_frames": []
            }
        }
    }

    try:
        runner = SDVideoPipelineRunner()
        runner.run(config)
        
        # 假设输出视频在输出目录中，查找生成的视频文件
        output_files = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
        if output_files:
            output_video = os.path.join(output_dir, output_files[0])
            return output_video
        else:
            raise gr.Error("未找到生成的输出视频")
        
    except Exception as e:
        raise gr.Error(f"处理过程中出现错误: {str(e)}")

def on_video_upload(video_path):
    """当视频上传时显示视频信息"""
    if video_path is None:
        return "请上传视频"
    
    try:
        orig_width, orig_height, width, height, fps, total_frames = get_video_info(video_path)
        duration = total_frames / fps if fps > 0 else 0
        
        info = f"""
        📹 视频信息:
        - 原始分辨率: {orig_width} × {orig_height}
        - 处理分辨率: {width} × {height}
        - 帧率: {fps:.2f} FPS
        - 总帧数: {total_frames}
        - 时长: {duration:.2f} 秒
        """
        
        if orig_width != width or orig_height != height:
            info += f"\n\n⚠️ 注意: 视频尺寸已自动调整为64的倍数"
            
        return info
    except Exception as e:
        return f"无法读取视频信息: {str(e)}"

# 创建 Gradio 界面
with gr.Blocks(title="视频风格转换") as demo:
    gr.Markdown("# 🎥 视频风格转换工具")
    gr.Markdown("使用 AnimateDiff 和 ControlNet 将视频转换为动漫风格")
    
    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="输入视频", sources=["upload"])
            
            # 显示视频信息
            video_info = gr.Textbox(
                label="视频信息",
                interactive=False,
                lines=6,
                value="请上传视频以查看信息"
            )
            
            # 视频上传时更新信息
            input_video.upload(
                fn=on_video_upload,
                inputs=input_video,
                outputs=video_info
            )
            input_video.clear(
                fn=lambda: "请上传视频",
                inputs=[],
                outputs=video_info
            )
            
            with gr.Accordion("提示词设置", open=True):
                prompt = gr.Textbox(
                    label="正面提示词",
                    value="best quality, perfect anime illustration, light, a girl is dancing, smile, solo",
                    lines=3
                )
                negative_prompt = gr.Textbox(
                    label="负面提示词",
                    value="verybadimagenegative_v1.3",
                    lines=2
                )
            
            with gr.Accordion("高级设置", open=False):
                with gr.Row():
                    cfg_scale = gr.Slider(
                        label="CFG Scale", minimum=1.0, maximum=20.0, value=7.0, step=0.1
                    )
                    denoising_strength = gr.Slider(
                        label="去噪强度", minimum=0.1, maximum=1.0, value=1.0, step=0.1
                    )
                
                with gr.Row():
                    num_inference_steps = gr.Slider(
                        label="推理步数", minimum=1, maximum=50, value=10, step=1
                    )
                    seed = gr.Number(label="随机种子", value=0)
                
                with gr.Row():
                    tile_scale = gr.Slider(
                        label="Tile ControlNet 强度", minimum=0.0, maximum=2.0, value=0.5, step=0.1
                    )
                    lineart_scale = gr.Slider(
                        label="Lineart ControlNet 强度", minimum=0.0, maximum=2.0, value=0.5, step=0.1
                    )
            
            submit_btn = gr.Button("开始处理", variant="primary")
        
        with gr.Column():
            output_video = gr.Video(label="输出视频")
    
    # 设置提交事件
    submit_btn.click(
        fn=process_video,
        inputs=[
            input_video,
            prompt,
            negative_prompt,
            cfg_scale,
            denoising_strength,
            num_inference_steps,
            seed,
            tile_scale,
            lineart_scale
        ],
        outputs=output_video
    )
    
    # 添加使用说明
    with gr.Accordion("使用说明", open=False):
        gr.Markdown("""
        ## 使用说明
        
        1. **上传视频**: 选择要处理的视频文件
        2. **设置提示词**: 
           - 正面提示词: 描述你想要的画面
           - 负面提示词: 描述不想要的元素
        3. **调整参数**:
           - CFG Scale: 控制提示词的重要性 (7-12推荐)
           - 去噪强度: 控制风格化程度 (0.7-1.0推荐)
           - ControlNet 强度: 控制原视频结构的保持程度
        4. **开始处理**: 点击按钮开始转换
        
        ## 注意
        - 视频尺寸会自动调整为64的倍数以满足模型要求
        - 处理后的视频可能会与原始尺寸略有不同
        
        ## 硬件要求
        - GPU: 至少8GB显存
        - 内存: 至少16GB
        - 处理时间: 取决于视频长度和分辨率
        
        ## 访问地址
        - 本地访问: http://localhost:7860
        - 网络访问: http://[你的IP地址]:7860
        """)

if __name__ == "__main__":
    # 设置服务器IP和端口
    demo.launch(
        server_name="0.0.0.0",  # 允许所有网络接口访问
        server_port=7860,       # 设置端口为7860
        share=False             # 不创建gradio共享链接
    )