# Sample Geospatial Data for A2A World Platform

This directory contains sample geospatial data files for testing the enhanced data ingestion system.

## File Categories

### Sacred Sites and Cultural Landmarks
- **sacred_sites.kml** - KML file with famous sacred sites including Stonehenge, Pyramids, Machu Picchu
- **cultural_landmarks.geojson** - GeoJSON file with cultural landmarks from around the world
- **ancient_sites.csv** - CSV file with ancient archaeological sites and metadata

### Regional Collections
- **uk_sacred_sites.kml** - Sacred sites across the United Kingdom
- **european_monuments.geojson** - Historic monuments across Europe
- **world_heritage_sites.csv** - UNESCO World Heritage Sites data

### Test Cases
- **malformed_test.kml** - Intentionally malformed KML for error handling tests
- **invalid_coordinates.csv** - CSV with coordinate validation issues
- **empty_features.geojson** - GeoJSON with missing or invalid geometries

## Usage

These files can be used to:
1. Test the file upload and parsing functionality
2. Validate data quality assessment
3. Test error handling and recovery
4. Demonstrate various geospatial data formats
5. Performance testing with different file sizes

## File Formats Supported

- **KML/KMZ**: Google Earth format with placemarks, folders, and extended data
- **GeoJSON**: Standard geospatial JSON format with feature collections
- **CSV**: Comma-separated values with coordinate columns and metadata
- **ZIP**: Archives containing multiple geospatial files