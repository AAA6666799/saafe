import { useState } from "react";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/AppSidebar";
import HeliosMap from "@/components/HeliosMap";
import AthenaDashboard from "@/components/AthenaDashboard";
import AssetGrid from "@/components/AssetGrid";
import FireDetection from "@/components/FireDetection";
import CameraDetailsPanel from "@/components/CameraDetailsPanel";
import AIChatbot from "@/components/AIChatbot";
import ThemeToggle from "@/components/ThemeToggle";
import EmailRecipientManager from "@/components/EmailRecipientManager";
import AIAgentsConsensus from "@/components/AIAgentsConsensus";
import { API_BASE_URL } from "@/config/api";
// FireDataSender temporarily disabled for debugging
// import FireDataSender from "@/components/FireDataSender";

export interface Camera {
  id: string;
  name: string;
  location: string;
  coordinates: [number, number];
  status: "fire" | "no-fire" | "predicted" | "black";
  temperature: number;
  lastUpdated: string;
}

const Index = () => {
  const [selectedCamera, setSelectedCamera] = useState<Camera | null>(null);
  const [activeView, setActiveView] = useState<"helios" | "athena" | "grid" | "detection" | "email" | "agents">("helios");
  const [isChatOpen, setIsChatOpen] = useState(false);

  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full bg-background text-foreground">
        <AppSidebar activeView={activeView} setActiveView={setActiveView} />
        
        <div className="flex-1 flex flex-col">
          <header className="h-14 flex items-center justify-between border-b border-border bg-card px-4 sticky top-0 z-40">
            <div className="flex items-center">
              <SidebarTrigger className="mr-4" />
              <h1 className="text-lg font-semibold">Saafe Dashboard</h1>
            </div>
            <ThemeToggle />
          </header>

          <main className="flex-1 overflow-auto">
          {activeView === "helios" && (
            <HeliosMap onCameraSelect={setSelectedCamera} />
          )}
          {activeView === "athena" && (
            <AthenaDashboard />
          )}
          {activeView === "grid" && (
            <AssetGrid onCameraSelect={setSelectedCamera} />
          )}
          {activeView === "detection" && (
            <FireDetection onCameraSelect={setSelectedCamera} />
          )}
          {activeView === "agents" && (
            <div className="p-6">
              <AIAgentsConsensus apiBaseUrl={API_BASE_URL} />
            </div>
          )}
          {activeView === "email" && (
            <div className="p-6">
              <EmailRecipientManager apiBaseUrl={API_BASE_URL} />
              {/* FireDataSender component - temporarily disabled for debugging */}
              {/* <div className="mt-6">
                <FireDataSender />
              </div> */}
            </div>
          )}
          </main>
        </div>

        <CameraDetailsPanel 
          camera={selectedCamera} 
          onClose={() => setSelectedCamera(null)} 
        />

        <AIChatbot isOpen={isChatOpen} onToggle={() => setIsChatOpen(!isChatOpen)} />
      </div>
    </SidebarProvider>
  );
};

export default Index;
