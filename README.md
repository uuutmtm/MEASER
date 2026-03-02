
python pipeline.py --attack 


% quantization["awq","gptq","gguf-4bit","gguf-8bit"]
python pipeline.py --defense quantization["awq"] --model

python pipeline.py --defense prune["50"] --model 


python LLM_sanitizer/pipeline.py --model model/Llama-2-7b-chat-hf --attacks maleficnet --payload LLM_sanitizer/payloads/test_payload.bin --device auto



## saser：
python LLM_sanitizer/pipeline.py --model model/Llama-2-7b-chat-hf --defense quant_gptq_4 --device auto --attacks saser --general_layer model.layers.0 --robust_layer model.layers.1 --robust_type gptq_4

robust type:gguf_4, gguf_8, awq_4, gptq_4