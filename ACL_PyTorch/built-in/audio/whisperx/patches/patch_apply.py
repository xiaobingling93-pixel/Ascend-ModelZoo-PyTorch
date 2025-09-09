import os
import sys
import torchaudio
import funasr


def main():
    torchaudio_path = torchaudio.__path__
    funasr_path = funasr.__path__

    if torchaudio.__version__ != '2.5.1':
        sys.exit("Expect torchaudio==2.5.1")
    
    if funasr.__version__ != '1.2.7':
        sys.exit("Expect funasr==1.2.7")
    
    os.system(f'patch -p0 {torchaudio_path[0]}/compliance/kaldi.py kaldi.patch')
    os.system(f'patch -p0 {funasr_path[0]}/models/fsmn_vad_streaming/model.py vad_model.patch')
    os.system(f'patch -p0 {funasr_path[0]}/frontends/wav_frontend.py wav_frontend.patch')

if __name__ == '__main__':
    main()