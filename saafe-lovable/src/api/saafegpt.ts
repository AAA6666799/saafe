// Example API stub for SAAFEGPT (replace with your server route)
interface Message {
  role: string;
  content: string;
}

interface SaafeGPTResponse {
  message: Message;
}

export async function askSaafeGPT(messages: Message[]): Promise<SaafeGPTResponse> {
  // Replace with a real fetch to your backend
  await new Promise(r => setTimeout(r, 300));
  return { message: { role: 'assistant', content: 'Risk reduced after ventilation; keep monitoring for 10 minutes.' } };
}
