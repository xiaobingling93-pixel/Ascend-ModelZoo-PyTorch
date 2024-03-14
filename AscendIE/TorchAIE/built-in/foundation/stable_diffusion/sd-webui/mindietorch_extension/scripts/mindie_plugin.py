import logging
import gradio as gr
from basicsr.utils import get_root_logger
from modules import scripts
from replace_torch_aie import replace_torch_aie
from config import NpuConfig

def listen_change(choice):
    if choice == 'MindIE_torch':
        print("switch to MindIE_torch")
        replace_torch_aie()
        return

class TorchAscendIEPlugin(scripts.Script):
    
    def __init__(self) -> None:
        super().__init__()
        self.logger = get_root_logger()
        self.logger.info("import MindIEPlugin")
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
                npu_radio = gr.Radio(choices = ['None', 'MindIE_torch'], value = "None", label="Inference Engine choices")
                npu_radio.change(listen_change, inputs = npu_radio)

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