import { useState } from "react";
import { MessageCircle, X, Send } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface AIChatbotProps {
  isOpen: boolean;
  onToggle: () => void;
}

const AIChatbot = ({ isOpen, onToggle }: AIChatbotProps) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "Hello! I'm your FireWatch AI assistant. I can help you analyze camera data, understand fire predictions, and provide insights about your monitoring system. How can I assist you today?"
    }
  ]);
  const [input, setInput] = useState("");

  const handleSend = () => {
    if (!input.trim()) return;

    const userMessage: Message = { role: "user", content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput("");

    setTimeout(() => {
      const assistantMessage: Message = {
        role: "assistant",
        content: "I'm a demo assistant. To enable full AI capabilities with real-time analysis and predictions, connect to Lovable Cloud for intelligent fire detection insights."
      };
      setMessages(prev => [...prev, assistantMessage]);
    }, 1000);
  };

  return (
    <>
      {!isOpen && (
        <Button
          onClick={onToggle}
          className="fixed bottom-6 right-6 h-14 w-14 rounded-full shadow-glow-safe bg-gradient-safe hover:scale-110 transition-transform"
          size="icon"
        >
          <MessageCircle className="h-6 w-6" />
        </Button>
      )}

      {isOpen && (
        <Card className="fixed bottom-6 right-6 w-96 h-[500px] shadow-2xl border-border flex flex-col animate-in slide-in-from-bottom">
          <div className="p-4 border-b border-border flex items-center justify-between bg-gradient-safe">
            <div className="flex items-center gap-2">
              <MessageCircle className="h-5 w-5" />
              <h3 className="font-bold">AI Assistant</h3>
            </div>
            <Button variant="ghost" size="icon" onClick={onToggle}>
              <X className="h-4 w-4" />
            </Button>
          </div>

          <ScrollArea className="flex-1 p-4">
            <div className="space-y-4">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[80%] rounded-lg p-3 ${
                      message.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted"
                    }`}
                  >
                    <p className="text-sm">{message.content}</p>
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>

          <div className="p-4 border-t border-border">
            <div className="flex gap-2">
              <Input
                placeholder="Ask about fire detection..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && handleSend()}
              />
              <Button size="icon" onClick={handleSend}>
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </Card>
      )}
    </>
  );
};

export default AIChatbot;
