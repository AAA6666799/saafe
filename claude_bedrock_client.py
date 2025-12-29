import boto3
import json
import os

# Set your AWS region (e.g., 'us-east-1')

AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

# Updated Claude Sonnet 4 model ID for Bedrock
MODEL_ID = 'anthropic.claude-sonnet-4-20250514-v1:0'

# Initialize the Bedrock Runtime client
bedrock = boto3.client(
    'bedrock-runtime',
    region_name=AWS_REGION
)

def generate_claude_response(prompt, max_tokens=1024):
    body = {
        "prompt": prompt,
        "max_tokens_to_sample": max_tokens,
        "temperature": 0.2
    }
    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body)
    )
    result = json.loads(response['body'].read())
    return result.get('completion', result)

if __name__ == "__main__":
    user_prompt = input("Enter your prompt for Claude Sonnet 4: ")
    output = generate_claude_response(user_prompt)
    print("\nClaude Sonnet 4 Response:\n", output)
