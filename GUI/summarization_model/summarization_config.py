class Config:
    MODEL_NAME = "omarsabri8756/AraT5v2-XLSum-arabic-text-summarization"
    MAX_INPUT_LENGTH = 512
    MAX_OUTPUT_LENGTH = 192
    PADDING = "max_length"
    TRUNCATION = True
    NUM_BEAMS = 3
    MIN_INPUT_LENGTH = 10
    EARLY_STOPPING = True
    REPETITION_PENALTY = 3.0
    LENGTH_PENALTY = 2.0
    NO_REPEAT_NGRAM_SIZE = 3

    #DEVICE = "cuda" if torch.cuda.is_available() else "cpu"










