from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

en_text = """
A private-public joint investigation team will be organized to conduct thorough investigation on the stench-causing facilities in the entire Songdo area, and the team includes the fire department, the police department, the Ministry of Environment, the National Institute of Environmental Research, the Korea Environment Corporation, the Korea Gas Safety Corporation, the Incheon Research Institute of Public Health and Environment, the Korea Meteorological Administration, the Incheon Free Economic Zone Authority, and civilians.
"""
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")

# translate Hindi to French
tokenizer.src_lang = "en"
encoded_hi = tokenizer(en_text, return_tensors="pt")
generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id("ko"))
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)