import os
import argparse
from pydub import AudioSegment
from pydub.utils import make_chunks

parser = argparse.ArgumentParser(description='splits a audio file from folder')

def main(args):
    for file in os.listdir(args.folder):
        audio = AudioSegment.from_file(os.path.join(args.folder, file))
        length = args.seconds * 1000 # this is in miliseconds
        chunks = make_chunks(audio, length)
        names = []
        for i, chunk in enumerate(chunks):
            _name = file.split("/")[-1]
            name = "{}_{}.wav".format(i, _name)
            wav_path = os.path.join(args.save_path, name)
            chunk.export(wav_path, format="wav")

parser.add_argument("--seconds", metavar='seconds', type=int,default=None, required=True, help="enter how many seconds you want to slice it")
parser.add_argument("--folder", metavar = 'folder', type=str,default=None, required=True, help='enter your folder path')
parser.add_argument("--save_path", metavar='save_path', type=str,default=None, required=True, help='enter your save path')
args = parser.parse_args()

main(args)