## 帮助信息
### === Model Options ===
###  --encoder_model        encoder model, default: encoder
###  --decoder_model        decoder model, default: decoder
###  --bs           batch size, default: 1
### === Inference Options ===
###  --output_dir   output dir, default: output
### === Environment Options ===
###  --soc          soc version [Ascend310P?], default: Ascend310P3
### === Help Options ===
###  -h             print this message

help() {
    sed -rn 's/^### ?//;T;p;' "$0"
}

## 参数设置
GETOPT_ARGS=`getopt -o 'h' -al encoder_model:,decoder_model:,bs:,output_dir:,soc: -- "$@"`
eval set -- "$GETOPT_ARGS"
while [ -n "$1" ]
do
    case "$1" in
        -h) help; exit 0 ;;
        --encoder_model) encoder_model=$2; shift 2;;
        --decoder_model) decoder_model=$2; shift 2;;
        --bs) bs=$2; shift 2;;
        --output_dir) output_dir=$2; shift 2;;
        --soc) soc=$2; shift 2;;
        --) break ;;
    esac
done

if [[ -z $encoder_model ]]; then encoder_model=encoder; fi
if [[ -z $decoder_model ]]; then decoder_model=decoder; fi
if [[ -z $bs ]]; then bs=1; fi
if [[ -z $output_dir ]]; then output_dir=output; fi
if [[ -z $soc ]]; then echo "error: missing 1 required argument: 'soc'"; exit 1 ; fi


atc --framework=5 --input_format=ND --log=error --soc_version=${soc} \
    --model=${encoder_model}.onnx --output=${output_dir}/${encoder_model}_bs${bs} \
    --input_shape="mel:${bs},80,250~3000" \

atc --framework=5 --input_format=ND --log=error --soc_version=${soc} \
    --model=${decoder_model}.onnx --output=${output_dir}/${decoder_model}_bs${bs} \
    --input_shape="tokens:${bs},1~4;in_n_layer_self_k_cache:6,${bs},448,512;in_n_layer_self_v_cache:6,${bs},448,512;n_layer_cross_k:6,${bs},250~3000,512;n_layer_cross_v:6,${bs},250~3000,512;offset:1" \