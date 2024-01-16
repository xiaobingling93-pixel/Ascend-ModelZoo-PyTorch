import onnx

onnx_model = onnx.load('./rpn_head/model/glip_rpn_head.onnx')
for node in onnx_model.graph.node:
    if node.domain != '':
        node.domain = ''
while len(onnx_model.opset_import) > 1:
    onnx_model.opset_import.pop(1)
onnx.save(onnx_model, './rpn_head/model/glip_rpn_head_new.onnx')

