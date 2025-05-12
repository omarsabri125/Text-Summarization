import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from .model import build_tranformer_model
from .inference import decode_sequence
from .config import CONFIG
import warnings

# Optional: suppress other warnings from Python and TensorFlow/Keras
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

transformer = build_tranformer_model()
transformer.load_weights(CONFIG["model_weights_path"])


def Translate(text):
    print("completed transformer model loading")
    return decode_sequence(text, transformer)

print(Translate("i wish i could go to the concert, it looks so interesting and happy"))
