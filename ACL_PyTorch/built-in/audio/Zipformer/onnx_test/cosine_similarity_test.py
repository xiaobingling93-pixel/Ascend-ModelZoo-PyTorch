from scipy.spatial.distance import cosine

def read_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read().split()
        numbers = [float(num) for num in data]
    return numbers

def cosine_similarity(list1, list2):
    return 1 - cosine(list1, list2)

om_enc_path = "./OM/encoder.txt"
om_dec_path = "./OM/decoder.txt"
om_join_path = "./OM/joiner.txt"

onnx_enc_path = "./ONNX/encoder.txt"
onnx_dec_path = "./ONNX/decoder.txt"
onnx_join_path = "./ONNX/joiner.txt"

om_enc = read_file(om_enc_path)
om_dec = read_file(om_dec_path)
om_join = read_file(om_join_path)

onnx_enc = read_file(onnx_enc_path)
onnx_dec = read_file(onnx_dec_path)
onnx_join = read_file(onnx_join_path)

similarity = cosine_similarity(om_enc, onnx_enc)
print(f"Cosine Similarity OM_ONNX_ENC: {similarity}")
similarity = cosine_similarity(om_dec, onnx_dec)
print(f"Cosine Similarity OM_ONNX_DEC: {similarity}")
similarity = cosine_similarity(om_join, onnx_join)
print(f"Cosine Similarity OM_ONNX_JOIN: {similarity}")