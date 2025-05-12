# Load or define your Arabic summarization model here
from summarization_model.summary_inference import SummarizationModel
import time
def summarize_arabic_text(text):
    # Replace with actual summarization logic
    time.sleep(10)
    # Initialize the model (this should be done once, not every time you call the function)
    model = SummarizationModel()
    # Call the model's summarize method
    text = text.strip()
    summary = model.summarize(text)
    return summary
