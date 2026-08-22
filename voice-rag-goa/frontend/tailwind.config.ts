/**
 * Tailwind CSS v4 is CSS-first: the Goa palette and design tokens live in
 * `src/app/globals.css` under `@theme`. This file exists only to declare the
 * content globs (so class scanning is explicit) and is intentionally minimal.
 */
import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/app/**/*.{ts,tsx}",
    "./src/components/**/*.{ts,tsx}",
    "./src/lib/**/*.{ts,tsx}",
  ],
};

export default config;
