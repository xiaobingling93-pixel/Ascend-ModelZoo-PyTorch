import logging
import gradio as gr
from basicsr.utils import get_root_logger
from modules import scripts
from replace_onnx import replace_onnx
from config import NpuConfig

def listen_change(choice):
    if choice == 'ONNX':
        print("switch to ONNX")
        replace_onnx()
        return
    else:
        print("do nothing...")


class AscendIEPlugin(scripts.Script):
    
    def __init__(self) -> None:
        super().__init__()
        self.logger = get_root_logger()
        self.logger.info("import AscendIEPlugin")
        self.logger.setLevel(logging.INFO)
    
    def title(self):
        return "webui-npu-extension"
    
    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def ui(self, is_txt2img):
        with gr.Group():
            with gr.Accordion("npu-extensions", open = True):
                device_radio = gr.Radio(choices = ['None', 'Duo', 'A2'], value = "None", label="Ascend device: Duo is 310P3; A2 is 910B4")
                device_radio.change(self.listen_device_change, inputs=device_radio)

                npu_radio = gr.Radio(choices = ['None', 'ONNX'], value = "None", label="Ascend boosting choices")
                npu_radio.change(listen_change, inputs=npu_radio)

                parallel_inferencing_checkbox = gr.Checkbox(label = 'Use_Parallel_Inferencing', info="Do you want to use parallel inferencing?")
                parallel_inferencing_checkbox.change(self.listen_parallel_status, inputs = parallel_inferencing_checkbox)
    
    def listen_parallel_status(self, status):
        if status:
            self.logger.info("Start to use parallel inferencing")
            NpuConfig.use_parallel_inferencing = True
            NpuConfig.unet_session = False
        else:
            self.logger.info("Stop using parallel inferencing")
            NpuConfig.use_parallel_inferencing = False
            NpuConfig.unet_session = False
            if NpuConfig.unet_session_bg:
                print("stop unet_session_bg")
                NpuConfig.unet_session_bg.stop()
                NpuConfig.unet_session_bg = False

    def listen_device_change(self, choice):
        if choice == 'None':
            print("do not use npu, use cpu default.")
            NpuConfig.use_cpu = True
            NpuConfig.Duo = False
            NpuConfig.A2 = False
            return
        elif choice == 'Duo':
            print("use Duo...")
            NpuConfig.use_cpu = False
            NpuConfig.Duo = True
            NpuConfig.A2 = False
            return
        elif choice == 'A2':
            print("use A2...")
            NpuConfig.use_cpu = False
            NpuConfig.Duo = False
            NpuConfig.A2 = True
            return
