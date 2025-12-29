# Multi-stage build for SAAFE Dashboard
FROM node:18-alpine AS frontend-build

WORKDIR /app/frontend
COPY saafe-lovable/package*.json ./
RUN npm ci --only=production
COPY saafe-lovable/ ./
RUN npm run build

# Backend stage
FROM node:18-alpine

WORKDIR /app

# Install system dependencies
RUN apk add --no-cache curl

# Copy backend package files
COPY saafe-lovable/backend/package*.json ./
RUN npm ci --only=production

# Copy backend code
COPY saafe-lovable/backend/ ./

# Copy built frontend to be served by backend
COPY --from=frontend-build /app/frontend/dist ./dist

# Create non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S saafe -u 1001
USER saafe

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/fire-detection-data || exit 1

# Start command
CMD ["npm", "start"]