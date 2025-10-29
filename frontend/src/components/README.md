# A2A World Platform - Visualization Components

This directory contains comprehensive data visualization components that complete Phase 2 of the A2A World Platform. These components provide interactive charts, maps, dashboard widgets, and pattern visualization tools for analyzing cultural and geospatial data.

## 📊 Chart Components (`/charts`)

### Available Components

#### `BarChart.tsx`
- **Purpose**: Dataset statistics, pattern counts, and distribution analysis
- **Features**: Data labels, tooltips, responsive design, color customization
- **Use Cases**: Agent performance metrics, file type distribution, processing status

```typescript
import { BarChart } from '@/components/charts';

<BarChart
  data={[
    { name: 'KML Files', value: 25 },
    { name: 'GeoJSON', value: 18 },
    { name: 'CSV', value: 12 }
  ]}
  title="Dataset Distribution"
  showDataLabels={true}
/>
```

#### `LineChart.tsx`
- **Purpose**: Time series analysis and trend visualization
- **Features**: Multiple lines, area fill, reference lines, date formatting
- **Use Cases**: Pattern discovery trends, system metrics over time, temporal analysis

```typescript
import { LineChart } from '@/components/charts';

<LineChart
  data={timeSeriesData}
  lines={[
    { key: 'discovered', name: 'Patterns Discovered', color: '#3B82F6' },
    { key: 'validated', name: 'Patterns Validated', color: '#10B981' }
  ]}
  xAxisFormat="date"
  title="Pattern Discovery Trends"
/>
```

#### `PieChart.tsx`
- **Purpose**: Category distribution with percentages
- **Features**: Donut chart option, custom colors, interactive tooltips
- **Use Cases**: Data quality distribution, file types, system resource usage

#### `HistogramChart.tsx`
- **Purpose**: Data distribution analysis with statistical overlays
- **Features**: Configurable bins, mean/median indicators, statistical calculations
- **Use Cases**: Confidence score distribution, coordinate analysis

#### `ScatterPlot.tsx`
- **Purpose**: Relationship visualization and correlation analysis
- **Features**: Multiple series, trend lines, color/size encoding
- **Use Cases**: Spatial data relationships, pattern clustering, multi-dimensional analysis

## 🗺️ Map Components (`/maps`)

### `ChoroplethMap.tsx`
- **Purpose**: Regional data visualization with colored polygons
- **Features**: Color scales, interactive tooltips, automatic bounds fitting
- **Use Cases**: Pattern density by region, statistical significance mapping

```typescript
import { ChoroplethMap } from '@/components/maps';

<ChoroplethMap
  data={regionData}
  colorScale={{
    min: 0,
    max: 100,
    colors: ['#FEF0D9', '#FDCC8A', '#FC8D59', '#E34A33', '#B30000']
  }}
  title="Pattern Density by Region"
/>
```

## 📈 Dashboard Widgets (`/widgets`)

### `StatCard.tsx`
- **Purpose**: Key metric display with trends and icons
- **Features**: Multiple sizes, trend indicators, color themes, click handlers
- **Use Cases**: KPI displays, system health indicators, summary statistics

```typescript
import { StatCard } from '@/components/widgets';

<StatCard
  title="Total Patterns"
  value={156}
  icon={Brain}
  color="purple"
  trend={{ value: 12.5, label: 'this week', direction: 'up' }}
/>
```

### `ProgressWidget.tsx`
- **Purpose**: Process status with step-by-step progress
- **Features**: Multi-step progress, pause/resume controls, error handling
- **Use Cases**: Data processing status, agent task progress, file upload status

## 🧠 Pattern Visualization (`/patterns`)

### `ConfidenceIndicator.tsx`
- **Purpose**: Pattern confidence visualization with circular progress
- **Features**: Threshold indicators, confidence levels, animated progress
- **Use Cases**: Pattern validation scores, algorithm confidence, quality metrics

```typescript
import { ConfidenceIndicator } from '@/components/patterns';

<ConfidenceIndicator
  value={0.847}
  threshold={0.7}
  size="lg"
  showThreshold={true}
  animated={true}
/>
```

## 🏗️ Dashboard Integration (`/dashboard`)

### `EnhancedDashboard.tsx`
- **Purpose**: Comprehensive dashboard integrating all visualization components
- **Features**: Auto-refresh, time range selection, responsive grid layout
- **Integration**: Connects to all API endpoints, real-time updates

## 📊 Sample Data (`/data`)

### `sampleData.ts`
- **Purpose**: Mock data generators for development and testing
- **Features**: Realistic data patterns, configurable parameters, comprehensive datasets

```typescript
import { SAMPLE_DATASETS, generateTimeSeriesData } from '@/data/sampleData';

// Use pre-generated sample data
const chartData = SAMPLE_DATASETS.timeSeries;

// Or generate custom data
const customData = generateTimeSeriesData(30, ['patterns', 'validations']);
```

## 🎨 Design System

### Color Palette
- **Primary Blue**: `#3B82F6` - Main UI elements, primary data series
- **Success Green**: `#10B981` - Success states, positive trends
- **Warning Yellow**: `#F59E0B` - Warnings, moderate confidence
- **Error Red**: `#EF4444` - Errors, low confidence, critical states
- **Purple**: `#8B5CF6` - Pattern-related elements, discovery indicators

### Responsive Breakpoints
- **Small**: `640px+` - Mobile devices
- **Medium**: `768px+` - Tablets
- **Large**: `1024px+` - Desktop
- **Extra Large**: `1280px+` - Large screens

## 🔧 Development Guidelines

### Component Structure
```
ComponentName/
├── ComponentName.tsx     # Main component
├── index.ts             # Export file
├── types.ts             # TypeScript interfaces
└── README.md            # Component documentation
```

### Props Interface Pattern
```typescript
export interface ComponentNameProps {
  // Required props
  data: DataType[];
  
  // Optional styling
  className?: string;
  height?: number;
  width?: number | string;
  
  // Feature flags
  loading?: boolean;
  error?: string;
  
  // Event handlers
  onClick?: (item: DataType) => void;
}
```

### Error Handling
All components include:
- Loading states with skeleton UI
- Error boundaries with retry options
- Graceful degradation for missing data
- Accessibility features (ARIA labels, keyboard navigation)

## 🚀 Usage Examples

### Basic Chart Integration
```typescript
import React, { useState, useEffect } from 'react';
import { BarChart, LineChart } from '@/components/charts';
import { dataApi } from '@/utils/api';

function DataDashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await dataApi.getDataSummary();
        setData(response.data);
      } catch (error) {
        console.error('Failed to fetch data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <BarChart
        data={data?.fileTypes || []}
        title="File Type Distribution"
        loading={loading}
      />
      <LineChart
        data={data?.trends || []}
        lines={[{ key: 'value', name: 'Trends', color: '#3B82F6' }]}
        title="Processing Trends"
        loading={loading}
      />
    </div>
  );
}
```

### Dashboard Widget Layout
```typescript
import { StatCard, ProgressWidget } from '@/components/widgets';
import { Activity, Database, Users } from 'lucide-react';

function SystemOverview() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      <StatCard
        title="Active Processes"
        value={12}
        icon={Activity}
        color="blue"
        trend={{ value: 8.5, label: 'vs last hour', direction: 'up' }}
      />
      <StatCard
        title="Data Sets"
        value={247}
        icon={Database}
        color="green"
      />
      <StatCard
        title="Online Agents"
        value={4}
        subtitle="of 5 total"
        icon={Users}
        color="purple"
      />
    </div>
  );
}
```

## 🔄 API Integration

### Standard Data Flow
1. **Fetch Data**: Use API client from `@/utils/api`
2. **Transform Data**: Convert API responses to component-expected format
3. **Handle States**: Manage loading, error, and success states
4. **Real-time Updates**: Use WebSocket connections for live data

### Sample API Integration
```typescript
import { patternsApi, agentsApi } from '@/utils/api';

// Fetch pattern statistics
const patternStats = await patternsApi.getStats();

// Transform for chart consumption
const chartData = patternStats.data.map(item => ({
  name: item.pattern_type,
  value: item.count,
  metadata: { confidence: item.avg_confidence }
}));
```

## 🎯 Phase 2 Completion Status

✅ **Completed Features:**
- Complete chart component library (Bar, Line, Pie, Histogram, Scatter)
- Enhanced map visualizations with choropleth support
- Dashboard widgets (StatCard, ProgressWidget)
- Pattern visualization components (ConfidenceIndicator)
- Comprehensive dashboard integration
- Sample data generators and mock datasets
- Responsive design and accessibility features
- Loading states and error handling
- TypeScript definitions and interfaces

🚀 **Ready for Production:**
- All components are production-ready
- Comprehensive error handling
- Performance optimized
- Accessible design
- Mobile responsive
- API integration ready

The A2A World Platform visualization system is now complete for Phase 2, providing a robust foundation for data analysis, pattern discovery visualization, and system monitoring.