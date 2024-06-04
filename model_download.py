from transformers import AutoModel, AutoTokenizer

model_name = "RUCKBReasoning/TableLLM-13b"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained("./TableLLM-13b")
tokenizer.save_pretrained("./TableLLM-13b")