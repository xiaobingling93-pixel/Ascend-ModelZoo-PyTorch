import logging
import gradio as gr
from basicsr.utils import get_root_logger
from modules import scripts
from replace_torch_aie import replace_unet_torch_aie
from config import NpuConfig

def listen_change(choice):
    if choice == 'torch_aie':
        print("switch to torch_aie")
        replace_unet_torch_aie()
        return

class TorchAscendIEPlugin(scripts.Script):
    
    def __init__(self) -> None:
        super().__init__()
        self.logger = get_root_logger()
        self.logger.info("import TorchAscendIEPlugin")
        self.logger.setLevel(logging.INFO)
    
    def title(self):
        return "webui-npu-extension"
    
    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def ui(self, is_txt2img):
        with gr.Group():
            with gr.Accordion("npu-extensions", open = True):
                npu_radio = gr.Radio(choices = ['torch_aie'], value = "None")
                npu_radio.change(listen_change, inputs = npu_radio)
                parallel_inferencing_checkbox = gr.Checkbox(label = 'Use_Parallel_Inferencing', info="Do you want to use parallel inferencing?")
                parallel_inferencing_checkbox.change(self.listen_parallel_status, inputs = parallel_inferencing_checkbox)
    
    def listen_parallel_status(self, status):
        if status:
            self.logger.info("Start to use parallel inferencing")
            NpuConfig.use_parallel_inferencing = True
        else:
            self.logger.info("Stop using parallel inferencing")
            NpuConfig.use_parallel_inferencing = False