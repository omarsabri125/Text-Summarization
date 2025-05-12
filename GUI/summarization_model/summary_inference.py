import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from summarization_model.summarization_config import Config
import sys
sys.stdout.reconfigure(encoding='utf-8')


class SummarizationModel:
    def __init__(self, model_name: str = Config.MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("Model loaded")

    def summarize(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", max_length=Config.MAX_INPUT_LENGTH,padding=Config.PADDING,
                                truncation=Config.TRUNCATION)
        
        summary_ids = self.model.generate(inputs["input_ids"],
                                          attention_mask=inputs["attention_mask"], 
                                          max_length=Config.MAX_OUTPUT_LENGTH, 
                                          num_beams=Config.NUM_BEAMS,
                                          min_length=Config.MIN_INPUT_LENGTH,
                                          repetition_penalty=Config.REPETITION_PENALTY,
                                          length_penalty=Config.LENGTH_PENALTY,
                                          no_repeat_ngram_size=Config.NO_REPEAT_NGRAM_SIZE,
                                          early_stopping=Config.EARLY_STOPPING
                                          )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

#summarization_model = SummarizationModel()

#text = "شهدت مدينة طرابلس، مساء أمس الأربعاء، احتجاجات شعبية وأعمال شغب لليوم الثالث على التوالي، وذلك بسبب تردي الوضع المعيشي والاقتصادي. واندلعت مواجهات عنيفة وعمليات كر وفر ما بين الجيش اللبناني والمحتجين استمرت لساعات، إثر محاولة فتح الطرقات المقطوعة، ما أدى إلى إصابة العشرات من الطرفين."
#summary=summarization_model.summarize(text)
#print(summary)
