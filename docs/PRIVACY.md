Default: OFFLINE_LLM=1; no images/meshes leave host.
Only de-identified numeric summaries sent if and only if --offline-llm=0.
Data flow:
[files] -> [validators] -> [ops] -> [LLM summaries? gated] -> [outputs]
Network guards: tests/test_offline_no_network.py
