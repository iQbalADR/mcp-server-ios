
import re

response_payload = """```swift
{"name": "write_to_file", "arguments": {"path": "/Users/user65419/Documents/Development/AI/xcode-mcp-server/models/LoginViewController.swift", "content": "// content"}}
```"""

print(f"Payload: {response_payload}")

tool_match = re.search(r"```(?:\w+)?\s*(\{.*?\})\s*```", response_payload, re.DOTALL)
if tool_match:
    print("MATCH SEEN!")
    print(tool_match.group(1))
else:
    print("NO MATCH")
