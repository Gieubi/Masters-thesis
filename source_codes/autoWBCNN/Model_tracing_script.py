from arch import deep_wb_single_task
from arch import deep_wb_blocks
import os
import torch
import torch.nn as nn

device = torch.device('cpu')

PATH = "./models/net_awb_model.pth"
nnModel = deep_wb_single_task.deepWBnet().to(device)
nnModel.load_state_dict(torch.load(PATH, map_location=device))
quantized_model = torch.quantization.quantize_dynamic(nnModel, {nn.Conv2d, nn.Linear}, dtype=torch.qint8)
quantized_model.eval()
nnModel.eval()

example = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(quantized_model.to(torch.device(device)), example.to(torch.device(device)))
traced_model.to(device)

traced_model.save('models/traced_net_awb_model.zip')