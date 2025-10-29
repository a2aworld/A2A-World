# A2A World Platform - Frontend Web Interface

A comprehensive React/Next.js web interface for the A2A World pattern discovery platform. This interface provides interactive dashboards, maps, and management tools for exploring geospatial-cultural data patterns discovered by AI agents.

## 🌟 Features

### 📊 Dashboard Overview
- **System Health Monitoring**: Real-time system status and service health
- **Agent Activity**: Live monitoring of active agents and their performance
- **Recent Discoveries**: Timeline of newly discovered patterns
- **Quick Actions**: Fast access to common tasks and operations

### 🗺️ Interactive Maps
- **Full-Screen Mapping**: Interactive map with geospatial data visualization
- **Pattern Overlays**: Visual representation of discovered patterns with clustering
- **Sacred Site Markers**: Cultural and sacred site locations with detailed popups
- **Layer Management**: Toggle different data layers and apply filters
- **Search & Navigation**: Find and navigate to specific locations

### 🧠 Pattern Explorer
- **Advanced Search**: Full-text search across pattern names and descriptions
- **Multi-Filter System**: Filter by type, status, confidence, and cultural relevance
- **Detailed Views**: Comprehensive pattern information with validation status
- **Export Capabilities**: Export patterns in multiple formats (JSON, KML, GeoJSON)
- **Validation Tracking**: View consensus scores and validation progress

### 👥 Agent Management
- **Live Status Monitoring**: Real-time agent health and performance metrics
- **Task Queue Visualization**: View active, queued, and completed tasks
- **Resource Monitoring**: CPU and memory usage tracking
- **Agent Controls**: Start, stop, and restart agents
- **Configuration Management**: View and edit agent configurations
- **Detailed Logs**: Access agent logs with filtering and search

### 📁 Data Management
- **File Upload Interface**: Drag-and-drop upload for KML, GeoJSON, CSV files
- **Dataset Organization**: Manage and organize uploaded datasets
- **Quality Reports**: Automated data quality analysis and validation
- **Import Status**: Track processing status and view detailed reports
- **Metadata Management**: Comprehensive metadata tracking and editing

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ and npm/yarn
- Running A2A World backend API (FastAPI)
- PostgreSQL database with PostGIS extension
- NATS messaging system

### Installation

1. **Install Dependencies**
   ```bash
   cd frontend
   npm install
   # or
   yarn install
   ```

2. **Environment Configuration**
   Create a `.env.local` file:
   ```env
   NEXT_PUBLIC_API_URL=http://localhost:8000
   NEXT_PUBLIC_WS_URL=ws://localhost:8000
   ```

3. **Start Development Server**
   ```bash
   npm run dev
   # or
   yarn dev
   ```

4. **Open in Browser**
   Navigate to [http://localhost:3000](http://localhost:3000)

### Production Build

```bash
# Build for production
npm run build

# Start production server
npm start
```

## 🏗️ Architecture

### Technology Stack
- **Framework**: Next.js 14+ with App Router
- **Language**: TypeScript for type safety
- **Styling**: Tailwind CSS with custom design system
- **State Management**: React hooks and context
- **Data Fetching**: Axios with SWR for caching
- **Real-time**: Socket.IO for WebSocket connections
- **Maps**: Leaflet with React-Leaflet
- **Forms**: React Hook Form with Zod validation
- **File Upload**: React Dropzone with progress tracking

### Project Structure
```
frontend/
├── src/
│   ├── components/          # Reusable UI components
│   │   └── ui/             # Basic UI building blocks
│   ├── hooks/              # Custom React hooks
│   │   └── useWebSocket.ts # WebSocket management
│   ├── pages/              # Next.js pages
│   │   ├── index.tsx       # Dashboard
│   │   ├── maps.tsx        # Interactive maps
│   │   ├── patterns.tsx    # Pattern explorer
│   │   ├── agents.tsx      # Agent management
│   │   └── data.tsx        # Data management
│   ├── types/              # TypeScript type definitions
│   │   └── index.ts        # Core platform types
│   ├── utils/              # Utility functions
│   │   └── api.ts          # API client configuration
│   └── styles/             # Global styles
├── public/                 # Static assets
├── package.json            # Dependencies and scripts
├── next.config.js          # Next.js configuration
├── tailwind.config.js      # Tailwind CSS configuration
└── tsconfig.json          # TypeScript configuration
```

## 📖 Usage Guide

### Dashboard Navigation
1. **System Overview**: Monitor overall system health and performance
2. **Quick Actions**: Access frequently used features via action cards
3. **Real-time Updates**: View live system status and recent discoveries

### Interactive Maps
1. **Layer Control**: Use the layers panel to toggle different data types
2. **Search Locations**: Use the search bar to find specific places
3. **Pattern Visualization**: Click on pattern overlays to view details
4. **Marker Information**: Click sacred site markers for detailed information

### Pattern Exploration
1. **Search Patterns**: Use the search bar for full-text pattern search
2. **Apply Filters**: Filter by type, status, confidence levels, and more
3. **Sort Results**: Sort by date, confidence, or cultural relevance
4. **View Details**: Click on any pattern to see comprehensive information
5. **Export Data**: Use export buttons to download pattern data

### Agent Management
1. **Monitor Status**: View real-time agent health and performance
2. **Control Agents**: Start, stop, or restart agents as needed
3. **View Tasks**: Monitor task queues and processing status
4. **Access Logs**: Review agent logs for troubleshooting
5. **Configuration**: View and modify agent settings

### Data Upload & Management
1. **Upload Files**: Drag files to the upload area or click to select
2. **Monitor Progress**: Track upload and processing status
3. **Quality Review**: Review automated quality reports
4. **Manage Metadata**: Edit dataset information and tags
5. **Export/Download**: Export processed datasets

## 🔧 Configuration

### Environment Variables
- `NEXT_PUBLIC_API_URL`: Backend API base URL
- `NEXT_PUBLIC_WS_URL`: WebSocket server URL
- `NEXT_PUBLIC_MAP_TILES_URL`: Custom map tiles URL (optional)

### API Integration
The frontend integrates with the A2A World FastAPI backend through:
- RESTful API endpoints for data operations
- WebSocket connections for real-time updates
- File upload endpoints for dataset management

### Customization
- **Branding**: Update colors and branding in [`tailwind.config.js`](tailwind.config.js)
- **API Endpoints**: Configure API URLs in [`src/utils/api.ts`](src/utils/api.ts)
- **Map Settings**: Customize map options in [`src/pages/maps.tsx`](src/pages/maps.tsx)

## 🌐 Browser Support
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## 📱 Responsive Design
The interface is fully responsive and optimized for:
- **Desktop**: Full-featured interface with all capabilities
- **Tablet**: Optimized layouts with touch-friendly controls
- **Mobile**: Essential features with streamlined navigation

## 🔒 Security Considerations
- API authentication using JWT tokens
- CSRF protection for form submissions
- Content Security Policy (CSP) headers
- Secure file upload validation
- Environment variable protection

## 🐛 Troubleshooting

### Common Issues

**TypeScript Errors**: 
- Install dependencies: `npm install`
- Most TypeScript errors are due to missing dependencies

**Map Not Loading**:
- Check internet connection for tile downloads
- Verify API URLs in environment configuration
- Ensure geospatial data has valid coordinates

**Real-time Updates Not Working**:
- Verify WebSocket URL configuration
- Check if backend WebSocket server is running
- Review browser console for connection errors

**File Upload Failures**:
- Check file size limits (configurable in backend)
- Ensure supported file formats (KML, GeoJSON, CSV, ZIP)
- Verify backend storage configuration

### Debug Mode
Enable debug mode by setting:
```env
NODE_ENV=development
NEXT_PUBLIC_DEBUG=true
```

## 🤝 Contributing
1. Follow TypeScript best practices
2. Use Tailwind CSS for styling
3. Implement responsive design for all components
4. Add proper error handling and loading states
5. Include accessibility features (ARIA labels, keyboard navigation)

## 📄 License
This project is part of the A2A World platform. See the main project README for license information.

## 🔗 Related Documentation
- [A2A World Main Documentation](../README.md)
- [API Documentation](../api/README.md)
- [Agent System Documentation](../agents/README.md)
- [Database Schema Documentation](../database/README.md)

---

**Built with ❤️ for the A2A World Pattern Discovery Platform**