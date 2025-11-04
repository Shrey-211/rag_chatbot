# RAG Chatbot Web Dashboard

A modern, dark-themed React frontend for the RAG Chatbot API built with Vite, TypeScript, and React 18.

## Features

- 📤 **Document Upload**: Index text snippets or upload files (PDF, DOCX, TXT, CSV)
- 📊 **Real-time Stats**: Monitor indexed documents, vector store, and provider status
- 💬 **Interactive Chat**: Query your knowledge base with cited sources
- 🎨 **Modern Dark UI**: GitHub-inspired dark theme with smooth animations
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile

## Prerequisites

- Node.js 18+ and npm
- RAG Chatbot API running on `http://localhost:8000` (or configure via environment variable)

## Quick Start

```bash
# Install dependencies
npm install

# Copy environment template (optional)
cp env.example .env

# Start development server
npm run dev
```

The app will be available at `http://localhost:5173`.

## Environment Configuration

Create a `.env` file to customize the API endpoint:

```env
VITE_API_BASE_URL=http://localhost:8000
```

If not set, the app defaults to `http://localhost:8000`.

## Build for Production

```bash
# Build the app
npm run build

# Preview the production build locally
npm run preview
```

The production-ready files will be in the `dist/` directory.

## Deployment

### Static Hosting (Vercel, Netlify, etc.)

1. Build the app: `npm run build`
2. Deploy the `dist/` folder to your hosting provider
3. Set the `VITE_API_BASE_URL` environment variable to your production API URL
4. Ensure your API has CORS configured to allow requests from your frontend domain

### Docker

Add a multi-stage Dockerfile:

```dockerfile
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## Project Structure

```
webapp/
├── public/
│   └── favicon.svg          # App icon
├── src/
│   ├── App.tsx              # Main application component
│   ├── App.css              # Component styles (dark theme)
│   ├── index.css            # Global styles
│   └── main.tsx             # App entry point
├── index.html               # HTML template
├── package.json             # Dependencies
├── tsconfig.json            # TypeScript config
└── vite.config.ts           # Vite config
```

## API Integration

The app integrates with the following RAG Chatbot API endpoints:

- `GET /health` - Health check
- `GET /stats` - System statistics
- `POST /index` - Index text content
- `POST /index/file` - Upload and index files
- `POST /query` - Query the knowledge base

## Troubleshooting

### CORS Errors

If you see CORS errors in the browser console, ensure your API `config.yaml` includes:

```yaml
api:
  cors_origins:
    - "http://localhost:5173"
    - "http://127.0.0.1:5173"
```

Then restart your API server.

### API Connection Issues

1. Verify the API is running: `curl http://localhost:8000/health`
2. Check the `VITE_API_BASE_URL` environment variable
3. Look for network errors in the browser DevTools console

## Tech Stack

- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **CSS3** - Styling with modern features (Grid, Flexbox, gradients)

## License

MIT

