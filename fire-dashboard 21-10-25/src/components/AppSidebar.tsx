import { Globe, LayoutDashboard, Grid3x3, Flame, Mail, Bot, Menu } from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarTrigger,
} from "@/components/ui/sidebar";

interface AppSidebarProps {
  activeView: string;
  setActiveView: (view: "helios" | "athena" | "grid" | "detection" | "email" | "agents") => void;
}

const menuItems = [
  { id: "helios", title: "Helios", icon: Globe, label: "Global View" },
  { id: "athena", title: "Athena", icon: LayoutDashboard, label: "Strategic Dashboard" },
  { id: "grid", title: "Grid", icon: Grid3x3, label: "Asset Manager" },
  { id: "detection", title: "Fire Prediction", icon: Flame, label: "Prediction System" },
  { id: "agents", title: "AI Agents", icon: Bot, label: "24 AI Consensus" },
  { id: "email", title: "Email Alerts", icon: Mail, label: "Email Configuration" },
];

export function AppSidebar({ activeView, setActiveView }: AppSidebarProps) {
  return (
    <Sidebar className="border-r border-border">
      <div className="p-4 border-b border-border flex items-center justify-between">
        <h1 className="text-xl font-bold bg-gradient-fire bg-clip-text text-transparent">
          SAAFE AI
        </h1>
        <SidebarTrigger>
          <Menu className="h-5 w-5" />
        </SidebarTrigger>
      </div>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Navigation</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {menuItems.map((item) => (
                <SidebarMenuItem key={item.id}>
                  <SidebarMenuButton
                    onClick={() => setActiveView(item.id as any)}
                    isActive={activeView === item.id}
                    className="w-full"
                  >
                    <item.icon className="h-4 w-4" />
                    <div className="flex flex-col items-start">
                      <span className="font-semibold">{item.title}</span>
                      <span className="text-xs text-muted-foreground">{item.label}</span>
                    </div>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
}
