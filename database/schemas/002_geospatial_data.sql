-- A2A World Platform - Geospatial Data Schema
-- PostGIS spatial data tables with geometry support and spatial indexes

SET search_path TO a2a_world, public;

-- Sacred sites and cultural landmarks
CREATE TABLE sacred_sites (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    site_type VARCHAR(100) CHECK (site_type IN (
        'temple', 'shrine', 'monument', 'burial_ground', 'ceremonial_site',
        'pilgrimage_site', 'natural_sacred', 'archaeological', 'historical'
    )),
    culture VARCHAR(100),
    time_period VARCHAR(100),
    location GEOMETRY(POINT, 4326) NOT NULL,
    elevation_meters DECIMAL(10,2),
    significance_level INTEGER CHECK (significance_level BETWEEN 1 AND 5),
    verified BOOLEAN DEFAULT FALSE,
    source_reference TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Geospatial features from imported data (KML, GeoJSON, etc.)
CREATE TABLE geospatial_features (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id UUID REFERENCES datasets(id) ON DELETE CASCADE,
    name VARCHAR(255),
    description TEXT,
    geometry GEOMETRY(GEOMETRY, 4326) NOT NULL,
    properties JSONB,
    feature_type VARCHAR(100),
    source_layer VARCHAR(255),
    style_info JSONB, -- Store styling information from KML/GeoJSON
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Geographic regions and boundaries
CREATE TABLE geographic_regions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    region_type VARCHAR(100) CHECK (region_type IN (
        'country', 'state', 'province', 'city', 'cultural_region',
        'watershed', 'mountain_range', 'desert', 'forest', 'coastline'
    )),
    boundary GEOMETRY(MULTIPOLYGON, 4326),
    center_point GEOMETRY(POINT, 4326),
    area_sqkm DECIMAL(15,2),
    population BIGINT,
    administrative_level INTEGER,
    parent_region_id UUID REFERENCES geographic_regions(id) ON DELETE SET NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Environmental data points (climate, seismic, etc.)
CREATE TABLE environmental_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    location GEOMETRY(POINT, 4326) NOT NULL,
    data_type VARCHAR(100) CHECK (data_type IN (
        'temperature', 'precipitation', 'humidity', 'wind_speed', 'atmospheric_pressure',
        'seismic_activity', 'magnetic_field', 'solar_radiation', 'air_quality',
        'vegetation_index', 'soil_composition', 'water_quality'
    )),
    measurement_value DECIMAL(15,6),
    measurement_unit VARCHAR(50),
    measurement_date TIMESTAMP WITH TIME ZONE NOT NULL,
    measurement_duration_hours INTEGER DEFAULT 0,
    quality_score DECIMAL(3,2) CHECK (quality_score BETWEEN 0 AND 1),
    source VARCHAR(255),
    sensor_id VARCHAR(100),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Environmental time series data (partitioned by month for performance)
CREATE TABLE environmental_time_series (
    id UUID DEFAULT uuid_generate_v4(),
    location_id UUID, -- Reference to a location point
    location GEOMETRY(POINT, 4326) NOT NULL,
    data_type VARCHAR(100) NOT NULL,
    timestamp_utc TIMESTAMP WITH TIME ZONE NOT NULL,
    value DECIMAL(15,6),
    unit VARCHAR(50),
    source VARCHAR(255),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (id, timestamp_utc)
) PARTITION BY RANGE (timestamp_utc);

-- Ley lines and energy grid data
CREATE TABLE ley_lines (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255),
    description TEXT,
    line_geometry GEOMETRY(LINESTRING, 4326) NOT NULL,
    strength_rating DECIMAL(3,2) CHECK (strength_rating BETWEEN 0 AND 10),
    discovery_method VARCHAR(100),
    validation_status VARCHAR(50) DEFAULT 'unverified' CHECK (validation_status IN (
        'unverified', 'pending_validation', 'validated', 'disputed', 'rejected'
    )),
    connected_sites UUID[], -- Array of sacred_sites IDs
    researcher_notes TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Geological formations and features
CREATE TABLE geological_features (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    feature_type VARCHAR(100) CHECK (feature_type IN (
        'mountain', 'volcano', 'cave', 'spring', 'fault_line', 'crater',
        'canyon', 'plateau', 'ridge', 'valley', 'cliff', 'rock_formation'
    )),
    geometry GEOMETRY(GEOMETRY, 4326) NOT NULL,
    elevation_meters DECIMAL(10,2),
    geological_age VARCHAR(100),
    rock_type VARCHAR(100),
    formation_process VARCHAR(255),
    cultural_significance TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Astronomical alignments and celestial correlations
CREATE TABLE astronomical_alignments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    site_id UUID REFERENCES sacred_sites(id) ON DELETE CASCADE,
    alignment_type VARCHAR(100) CHECK (alignment_type IN (
        'solar_solstice', 'solar_equinox', 'lunar_standstill', 'star_alignment',
        'constellation_alignment', 'planetary_alignment', 'eclipse_alignment'
    )),
    celestial_body VARCHAR(100),
    alignment_direction GEOMETRY(LINESTRING, 4326),
    azimuth_degrees DECIMAL(6,3) CHECK (azimuth_degrees >= 0 AND azimuth_degrees < 360),
    elevation_degrees DECIMAL(5,2) CHECK (elevation_degrees >= -90 AND elevation_degrees <= 90),
    alignment_date DATE,
    precision_arc_seconds DECIMAL(8,2),
    verification_status VARCHAR(50) DEFAULT 'unverified',
    researcher_notes TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create spatial indexes for all geometry columns
CREATE INDEX idx_sacred_sites_location ON sacred_sites USING GIST (location);
CREATE INDEX idx_sacred_sites_site_type ON sacred_sites(site_type);
CREATE INDEX idx_sacred_sites_culture ON sacred_sites(culture);
CREATE INDEX idx_sacred_sites_significance ON sacred_sites(significance_level);

CREATE INDEX idx_geospatial_features_geometry ON geospatial_features USING GIST (geometry);
CREATE INDEX idx_geospatial_features_dataset ON geospatial_features(dataset_id);
CREATE INDEX idx_geospatial_features_type ON geospatial_features(feature_type);

CREATE INDEX idx_geographic_regions_boundary ON geographic_regions USING GIST (boundary);
CREATE INDEX idx_geographic_regions_center ON geographic_regions USING GIST (center_point);
CREATE INDEX idx_geographic_regions_type ON geographic_regions(region_type);
CREATE INDEX idx_geographic_regions_parent ON geographic_regions(parent_region_id);

CREATE INDEX idx_environmental_data_location ON environmental_data USING GIST (location);
CREATE INDEX idx_environmental_data_type ON environmental_data(data_type);
CREATE INDEX idx_environmental_data_date ON environmental_data(measurement_date);
CREATE INDEX idx_environmental_data_source ON environmental_data(source);

CREATE INDEX idx_ley_lines_geometry ON ley_lines USING GIST (line_geometry);
CREATE INDEX idx_ley_lines_validation ON ley_lines(validation_status);
CREATE INDEX idx_ley_lines_strength ON ley_lines(strength_rating);

CREATE INDEX idx_geological_features_geometry ON geological_features USING GIST (geometry);
CREATE INDEX idx_geological_features_type ON geological_features(feature_type);
CREATE INDEX idx_geological_features_elevation ON geological_features(elevation_meters);

CREATE INDEX idx_astronomical_alignments_site ON astronomical_alignments(site_id);
CREATE INDEX idx_astronomical_alignments_direction ON astronomical_alignments USING GIST (alignment_direction);
CREATE INDEX idx_astronomical_alignments_type ON astronomical_alignments(alignment_type);
CREATE INDEX idx_astronomical_alignments_date ON astronomical_alignments(alignment_date);

-- Create GIN indexes for JSONB metadata fields
CREATE INDEX idx_sacred_sites_metadata_gin ON sacred_sites USING gin(metadata);
CREATE INDEX idx_geospatial_features_properties_gin ON geospatial_features USING gin(properties);
CREATE INDEX idx_geospatial_features_style_gin ON geospatial_features USING gin(style_info);
CREATE INDEX idx_geographic_regions_metadata_gin ON geographic_regions USING gin(metadata);
CREATE INDEX idx_environmental_data_metadata_gin ON environmental_data USING gin(metadata);
CREATE INDEX idx_ley_lines_metadata_gin ON ley_lines USING gin(metadata);
CREATE INDEX idx_geological_features_metadata_gin ON geological_features USING gin(metadata);
CREATE INDEX idx_astronomical_alignments_metadata_gin ON astronomical_alignments USING gin(metadata);

-- Create array indexes for ley_lines connected_sites
CREATE INDEX idx_ley_lines_connected_sites_gin ON ley_lines USING gin(connected_sites);

-- Apply updated_at triggers
CREATE TRIGGER update_sacred_sites_updated_at 
    BEFORE UPDATE ON sacred_sites 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_geographic_regions_updated_at 
    BEFORE UPDATE ON geographic_regions 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ley_lines_updated_at 
    BEFORE UPDATE ON ley_lines 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_geological_features_updated_at 
    BEFORE UPDATE ON geological_features 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_astronomical_alignments_updated_at 
    BEFORE UPDATE ON astronomical_alignments 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create initial partitions for environmental time series (current and next 6 months)
SELECT create_monthly_partition('environmental_time_series', date_trunc('month', CURRENT_DATE)::date);
SELECT create_monthly_partition('environmental_time_series', date_trunc('month', CURRENT_DATE + interval '1 month')::date);
SELECT create_monthly_partition('environmental_time_series', date_trunc('month', CURRENT_DATE + interval '2 months')::date);
SELECT create_monthly_partition('environmental_time_series', date_trunc('month', CURRENT_DATE + interval '3 months')::date);
SELECT create_monthly_partition('environmental_time_series', date_trunc('month', CURRENT_DATE + interval '4 months')::date);
SELECT create_monthly_partition('environmental_time_series', date_trunc('month', CURRENT_DATE + interval '5 months')::date);
SELECT create_monthly_partition('environmental_time_series', date_trunc('month', CURRENT_DATE + interval '6 months')::date);

-- Create constraint to ensure environmental time series location consistency
ALTER TABLE environmental_time_series 
ADD CONSTRAINT environmental_time_series_location_not_null 
CHECK (location IS NOT NULL);