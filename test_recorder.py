from src.attention_recorder import AttentionRecorder

# 初始化
recorder = AttentionRecorder(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    top_k=100
)

# 测试记录
prompt = "Read the file /home/user/code.py and explain what it does."
response = "The file contains a Python function that..."
tool_calls = [{"tool": "file_read", "arguments": {"path": "/home/user/code.py"}}]

record = recorder.record(
    prompt=prompt,
    response=response,
    tool_calls=tool_calls,
    task_id="test-001",
    round_id=0
)

# 保存
recorder.save_to_json(record, "results/test_record.json")
print(f"✅ Test passed! Recorded {len(record.top_k_scores)} tokens")
