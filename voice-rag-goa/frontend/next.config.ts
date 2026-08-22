import type { NextConfig } from "next";

/**
 * The frontend talks to the FastAPI backend over both REST and a WebSocket.
 * In development we let the browser hit the backend origin directly
 * (NEXT_PUBLIC_API_BASE, default http://localhost:8000) so the WebSocket
 * upgrade is not proxied. `reactStrictMode` is off because it double-invokes
 * effects, which would open the audio WebSocket twice and duplicate the mic
 * stream in dev.
 *
 * Note: Next.js 16 removed the built-in lint step from `next build` (and the
 * `eslint` config key), so there is nothing to disable here — run
 * `npm run lint` explicitly when you want the report.
 */
const nextConfig: NextConfig = {
  reactStrictMode: false,
};

export default nextConfig;
