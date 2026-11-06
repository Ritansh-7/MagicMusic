import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = { title: "CozyBeats", description: "Your sound, evolving." };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
