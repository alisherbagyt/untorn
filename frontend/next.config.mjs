/** @type {import('next').NextConfig} */
const nextConfig = {
  // Proxy API calls to the FastAPI backend
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.BACKEND_URL || "http://localhost:8000"}/:path*`,
      },
    ];
  },
  // Disable image optimization for local files served from backend
  images: {
    remotePatterns: [
      { protocol: "http", hostname: "localhost", port: "8000" },
      { protocol: "http", hostname: "127.0.0.1", port: "8000" },
    ],
    unoptimized: true,
  },
};

export default nextConfig;
