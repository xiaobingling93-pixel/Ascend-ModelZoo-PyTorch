# Convert YOLOv9-S

```
import torch
from models.yolo import Model

device = torch.device("cpu")
cfg = "./models/detect/gelan-s.yaml"
model = Model(cfg, ch=3, nc=80, anchors=3)
#model = model.half()
model = model.to(device)
_ = model.eval()
ckpt = torch.load('./yolov9-s.pt', map_location='cpu')
model.names = ckpt['model'].names
model.nc = ckpt['model'].nc

idx = 0
for k, v in model.state_dict().items():
    if "model.{}.".format(idx) in k:
        if idx < 22:
            kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
        elif "model.{}.cv2.".format(idx) in k:
            kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+7))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
        elif "model.{}.cv3.".format(idx) in k:
            kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+7))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
        elif "model.{}.dfl.".format(idx) in k:
            kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+7))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
    else:
        while True:
            idx += 1
            if "model.{}.".format(idx) in k:
                break
        if idx < 22:
            kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
        elif "model.{}.cv2.".format(idx) in k:
            kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+7))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
        elif "model.{}.cv3.".format(idx) in k:
            kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+7))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
        elif "model.{}.dfl.".format(idx) in k:
            kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+7))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
_ = model.eval()

m_ckpt = {'model': model.half(),
          'optimizer': None,
          'best_fitness': None,
          'ema': None,
          'updates': None,
          'opt': None,
          'git': None,
          'date': None,
          'epoch': -1}
torch.save(m_ckpt, "./yolov9-s-converted.pt")
```


# Convert YOLOv9-M

```
import torch
from models.yolo import Model

device = torch.device("cpu")
cfg = "./models/detect/gelan-m.yaml"
model = Model(cfg, ch=3, nc=80, anchors=3)
#model = model.half()
model = model.to(device)
_ = model.eval()
ckpt = torch.load('./yolov9-m.pt', map_location='cpu')
model.names = ckpt['model'].names
model.nc = ckpt['model'].nc

idx = 0
for k, v in model.state_dict().items():
    if "model.{}.".format(idx) in k:
        if idx < 22:
            kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+1))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
        elif "model.{}.cv2.".format(idx) in k:
            kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+16))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
        elif "model.{}.cv3.".format(idx) in k:
            kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+16))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
        elif "model.{}.dfl.".format(idx) in k:
            kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+16))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
    else:
        while True:
            idx += 1
            if "model.{}.".format(idx) in k:
                break
        if idx < 22:
            kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+1))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
        elif "model.{}.cv2.".format(idx) in k:
            kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+16))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
        elif "model.{}.cv3.".format(idx) in k:
            kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+16))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
        elif "model.{}.dfl.".format(idx) in k:
            kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+16))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
_ = model.eval()

m_ckpt = {'model': model.half(),
          'optimizer': None,
          'best_fitness': None,
          'ema': None,
          'updates': None,
          'opt': None,
          'git': None,
          'date': None,
          'epoch': -1}
torch.save(m_ckpt, "./yolov9-m-converted.pt")
```


# Convert YOLOv9-C

```
import torch
from models.yolo import Model

device = torch.device("cpu")
cfg = "./models/detect/gelan-c.yaml"
model = Model(cfg, ch=3, nc=80, anchors=3)
#model = model.half()
model = model.to(device)
_ = model.eval()
ckpt = torch.load('./yolov9-c.pt', map_location='cpu')
model.names = ckpt['model'].names
model.nc = ckpt['model'].nc

idx = 0
for k, v in model.state_dict().items():
    if "model.{}.".format(idx) in k:
        if idx < 22:
            kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+1))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
        elif "model.{}.cv2.".format(idx) in k:
            kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+16))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
        elif "model.{}.cv3.".format(idx) in k:
            kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+16))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
        elif "model.{}.dfl.".format(idx) in k:
            kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+16))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
    else:
        while True:
            idx += 1
            if "model.{}.".format(idx) in k:
                break
        if idx < 22:
            kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+1))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
        elif "model.{}.cv2.".format(idx) in k:
            kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+16))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
        elif "model.{}.cv3.".format(idx) in k:
            kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+16))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
        elif "model.{}.dfl.".format(idx) in k:
            kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+16))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
_ = model.eval()

m_ckpt = {'model': model.half(),
          'optimizer': None,
          'best_fitness': None,
          'ema': None,
          'updates': None,
          'opt': None,
          'git': None,
          'date': None,
          'epoch': -1}
torch.save(m_ckpt, "./yolov9-c-converted.pt")
```


# Convert YOLOv9-E

```
import torch
from models.yolo import Model

device = torch.device("cpu")
cfg = "./models/detect/gelan-e.yaml"
model = Model(cfg, ch=3, nc=80, anchors=3)
#model = model.half()
model = model.to(device)
_ = model.eval()
ckpt = torch.load('./yolov9-e.pt', map_location='cpu')
model.names = ckpt['model'].names
model.nc = ckpt['model'].nc

idx = 0
for k, v in model.state_dict().items():
    if "model.{}.".format(idx) in k:
        if idx < 29:
            kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
        elif idx < 42:
            kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+7))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
        elif "model.{}.cv2.".format(idx) in k:
            kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+7))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
        elif "model.{}.cv3.".format(idx) in k:
            kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+7))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
        elif "model.{}.dfl.".format(idx) in k:
            kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+7))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
    else:
        while True:
            idx += 1
            if "model.{}.".format(idx) in k:
                break
        if idx < 29:
            kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
        elif idx < 42:
            kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+7))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
        elif "model.{}.cv2.".format(idx) in k:
            kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+7))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
        elif "model.{}.cv3.".format(idx) in k:
            kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+7))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
        elif "model.{}.dfl.".format(idx) in k:
            kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+7))
            model.state_dict()[k] -= model.state_dict()[k]
            model.state_dict()[k] += ckpt['model'].state_dict()[kr]
            print(k, "perfectly matched!!")
_ = model.eval()

m_ckpt = {'model': model.half(),
          'optimizer': None,
          'best_fitness': None,
          'ema': None,
          'updates': None,
          'opt': None,
          'git': None,
          'date': None,
          'epoch': -1}
torch.save(m_ckpt, "./yolov9-e-converted.pt")
```
