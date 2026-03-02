# Code for "MEASER: Malware embedding attacks on open-source LLMs"
python pipeline.py --model model/Llama-2-7b-chat-hf --attacks measer --payload payloads/test_payload.bin --device cuda:0
