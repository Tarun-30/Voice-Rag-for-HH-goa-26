import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Voice RAG · Goa Hacker House",
  description:
    "Real-time voice-to-answer retrieval-augmented generation over MSMARCO-XI — " +
    "ultra-low-latency hybrid search, multi-strategy chunking, streaming grounded answers, " +
    "and full latency telemetry.",
  applicationName: "Voice RAG Goa",
};

export const viewport: Viewport = {
  themeColor: "#0a4d2e",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
