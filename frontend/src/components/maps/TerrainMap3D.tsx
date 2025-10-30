/**
 * A2A World Platform - 3D Terrain Map Component
 *
 * Interactive 3D terrain visualization using Three.js for elevation-based pattern visualization.
 * Supports real-time elevation data, pattern overlays, and interactive exploration.
 */

import React, { useRef, useEffect, useState, useMemo } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { clsx } from 'clsx';

export interface TerrainData {
  id: string;
  coordinates: [number, number]; // [lat, lng]
  elevation: number;
  properties: {
    name?: string;
    pattern_type?: string;
    confidence?: number;
    cultural_relevance?: number;
    [key: string]: any;
  };
}

export interface TerrainMap3DProps {
  data: TerrainData[];
  title?: string;
  subtitle?: string;
  height?: number;
  width?: string;
  center?: [number, number]; // [lat, lng]
  zoom?: number;
  elevationScale?: number;
  showPatterns?: boolean;
  showGrid?: boolean;
  className?: string;
  loading?: boolean;
  error?: string;
  onPointClick?: (point: TerrainData) => void;
  onPointHover?: (point: TerrainData | null) => void;
}

const DEFAULT_CENTER: [number, number] = [39.8283, -98.5795]; // Center of US

export function TerrainMap3D({
  data,
  title,
  subtitle,
  height = 600,
  width = '100%',
  center = DEFAULT_CENTER,
  zoom = 1,
  elevationScale = 0.1,
  showPatterns = true,
  showGrid = true,
  className,
  loading = false,
  error,
  onPointClick,
  onPointHover
}: TerrainMap3DProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene>();
  const rendererRef = useRef<THREE.WebGLRenderer>();
  const cameraRef = useRef<THREE.PerspectiveCamera>();
  const controlsRef = useRef<OrbitControls>();
  const animationFrameRef = useRef<number>();

  const [hoveredPoint, setHoveredPoint] = useState<TerrainData | null>(null);

  // Convert lat/lng to 3D coordinates (simple Mercator-like projection)
  const latLngTo3D = useMemo(() => {
    return (lat: number, lng: number, elevation: number = 0): [number, number, number] => {
      const x = (lng - center[1]) * 111320 * Math.cos(lat * Math.PI / 180); // meters
      const z = (center[0] - lat) * 111320; // meters
      const y = elevation * elevationScale;
      return [x, y, z];
    };
  }, [center, elevationScale]);

  // Create terrain mesh from elevation data
  const createTerrainMesh = useMemo(() => {
    if (!data || data.length === 0) return null;

    const geometry = new THREE.PlaneGeometry(10000, 10000, 100, 100);
    const vertices = geometry.attributes.position.array as Float32Array;

    // Simple interpolation for terrain height
    for (let i = 0; i < vertices.length; i += 3) {
      const x = vertices[i];
      const z = vertices[i + 2];

      // Find nearest data points and interpolate elevation
      let totalWeight = 0;
      let weightedElevation = 0;

      data.forEach(point => {
        const [px, , pz] = latLngTo3D(point.coordinates[0], point.coordinates[1]);
        const distance = Math.sqrt((x - px) ** 2 + (z - pz) ** 2);
        if (distance < 1000) { // Influence radius
          const weight = 1 / (distance + 1);
          totalWeight += weight;
          weightedElevation += point.elevation * weight;
        }
      });

      if (totalWeight > 0) {
        vertices[i + 1] = (weightedElevation / totalWeight) * elevationScale;
      }
    }

    geometry.attributes.position.needsUpdate = true;
    geometry.computeVertexNormals();

    const material = new THREE.MeshLambertMaterial({
      color: 0x8B7355,
      wireframe: false,
      transparent: true,
      opacity: 0.8
    });

    return new THREE.Mesh(geometry, material);
  }, [data, latLngTo3D, elevationScale]);

  // Create pattern visualization points
  const createPatternPoints = useMemo(() => {
    if (!data || !showPatterns) return [];

    return data.map(point => {
      const [x, y, z] = latLngTo3D(point.coordinates[0], point.coordinates[1], point.elevation);

      // Color based on pattern type and confidence
      let color = 0xff0000; // default red
      if (point.properties.pattern_type === 'cultural') {
        color = 0x00ff00; // green
      } else if (point.properties.pattern_type === 'geometric') {
        color = 0x0000ff; // blue
      } else if (point.properties.pattern_type === 'temporal') {
        color = 0xffff00; // yellow
      }

      // Adjust brightness based on confidence
      const confidence = point.properties.confidence || 0.5;
      const intensity = Math.max(0.3, confidence);
      color = new THREE.Color(color).multiplyScalar(intensity).getHex();

      const geometry = new THREE.SphereGeometry(50, 8, 8);
      const material = new THREE.MeshBasicMaterial({ color });
      const sphere = new THREE.Mesh(geometry, material);
      sphere.position.set(x, y + 100, z); // Offset above terrain
      sphere.userData = { pointData: point };

      return sphere;
    });
  }, [data, showPatterns, latLngTo3D]);

  // Initialize Three.js scene
  useEffect(() => {
    if (!mountRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x87CEEB); // Sky blue
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(75, mountRef.current.clientWidth / mountRef.current.clientHeight, 0.1, 100000);
    camera.position.set(5000, 3000, 5000);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Controls setup
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false;
    controls.minDistance = 1000;
    controls.maxDistance = 50000;
    controls.maxPolarAngle = Math.PI / 2;
    controlsRef.current = controls;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5000, 10000, 5000);
    directionalLight.castShadow = true;
    directionalLight.shadow.mapSize.width = 2048;
    directionalLight.shadow.mapSize.height = 2048;
    scene.add(directionalLight);

    // Grid helper
    if (showGrid) {
      const gridHelper = new THREE.GridHelper(20000, 20, 0x000000, 0x444444);
      gridHelper.position.y = -50;
      scene.add(gridHelper);
    }

    // Add terrain mesh
    if (createTerrainMesh) {
      scene.add(createTerrainMesh);
    }

    // Add pattern points
    createPatternPoints.forEach(point => {
      scene.add(point);
    });

    // Raycaster for mouse interactions
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const onMouseMove = (event: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(createPatternPoints);

      if (intersects.length > 0) {
        const intersectedPoint = intersects[0].object;
        const pointData = intersectedPoint.userData.pointData as TerrainData;
        setHoveredPoint(pointData);
        if (onPointHover) onPointHover(pointData);
      } else {
        setHoveredPoint(null);
        if (onPointHover) onPointHover(null);
      }
    };

    const onClick = (event: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(createPatternPoints);

      if (intersects.length > 0) {
        const intersectedPoint = intersects[0].object;
        const pointData = intersectedPoint.userData.pointData as TerrainData;
        if (onPointClick) onPointClick(pointData);
      }
    };

    renderer.domElement.addEventListener('mousemove', onMouseMove);
    renderer.domElement.addEventListener('click', onClick);

    // Animation loop
    const animate = () => {
      animationFrameRef.current = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Handle resize
    const handleResize = () => {
      if (!mountRef.current) return;
      camera.aspect = mountRef.current.clientWidth / mountRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      window.removeEventListener('resize', handleResize);
      renderer.domElement.removeEventListener('mousemove', onMouseMove);
      renderer.domElement.removeEventListener('click', onClick);
      if (mountRef.current && renderer.domElement.parentNode === mountRef.current) {
        mountRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [createTerrainMesh, createPatternPoints, showGrid, onPointClick, onPointHover]);

  if (loading) {
    return (
      <div className={clsx('flex items-center justify-center border rounded-lg', className)} style={{ height, width }}>
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-2 text-sm text-gray-500">Loading 3D terrain...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={clsx('flex items-center justify-center border rounded-lg', className)} style={{ height, width }}>
        <div className="text-center">
          <div className="text-red-500 mb-2">
            <svg className="h-8 w-8 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <p className="text-sm text-gray-600">{error}</p>
        </div>
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className={clsx('flex items-center justify-center border rounded-lg', className)} style={{ height, width }}>
        <div className="text-center">
          <div className="text-gray-400 mb-2">
            <svg className="h-8 w-8 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
            </svg>
          </div>
          <p className="text-sm text-gray-600">No terrain data to display</p>
        </div>
      </div>
    );
  }

  return (
    <div className={clsx('w-full', className)}>
      {(title || subtitle) && (
        <div className="mb-4">
          {title && (
            <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
          )}
          {subtitle && (
            <p className="text-sm text-gray-600 mt-1">{subtitle}</p>
          )}
        </div>
      )}

      <div className="relative border rounded-lg overflow-hidden bg-gray-100" style={{ height, width }}>
        <div ref={mountRef} className="w-full h-full" />

        {/* Hover tooltip */}
        {hoveredPoint && (
          <div className="absolute top-4 left-4 bg-white p-3 rounded-lg shadow-lg z-10 max-w-xs">
            <h4 className="font-medium text-gray-900">{hoveredPoint.properties.name || 'Pattern Point'}</h4>
            <div className="mt-2 space-y-1 text-sm text-gray-600">
              <p>Coordinates: {hoveredPoint.coordinates[0].toFixed(4)}, {hoveredPoint.coordinates[1].toFixed(4)}</p>
              <p>Elevation: {hoveredPoint.elevation.toFixed(1)}m</p>
              {hoveredPoint.properties.confidence && (
                <p>Confidence: {(hoveredPoint.properties.confidence * 100).toFixed(1)}%</p>
              )}
              {hoveredPoint.properties.cultural_relevance && (
                <p>Cultural Relevance: {(hoveredPoint.properties.cultural_relevance * 100).toFixed(1)}%</p>
              )}
            </div>
          </div>
        )}

        {/* Legend */}
        <div className="absolute bottom-4 right-4 bg-white p-3 rounded-lg shadow-lg z-10">
          <h4 className="text-sm font-medium text-gray-900 mb-2">Pattern Types</h4>
          <div className="space-y-1">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-red-500"></div>
              <span className="text-xs text-gray-600">Environmental</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
              <span className="text-xs text-gray-600">Cultural</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-blue-500"></div>
              <span className="text-xs text-gray-600">Geometric</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
              <span className="text-xs text-gray-600">Temporal</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default TerrainMap3D;