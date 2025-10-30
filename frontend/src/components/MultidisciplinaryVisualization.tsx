/**
 * A2A World Platform - Interactive Multidisciplinary Visualization Component
 *
 * Network visualization showing cross-disciplinary connections between patterns,
 * cultural contexts, environmental factors, and validation results.
 */

import React, { useRef, useEffect, useState, useMemo } from 'react';
import * as d3 from 'd3';
import { clsx } from 'clsx';

export interface Node {
  id: string;
  label: string;
  type: 'pattern' | 'cultural' | 'environmental' | 'validation' | 'agent';
  group: string;
  size: number;
  properties: {
    confidence?: number;
    relevance?: number;
    connections?: number;
    [key: string]: any;
  };
}

export interface Link {
  source: string;
  target: string;
  strength: number;
  type: 'cultural' | 'environmental' | 'validation' | 'agent_interaction';
  properties?: Record<string, any>;
}

export interface MultidisciplinaryData {
  nodes: Node[];
  links: Link[];
}

export interface MultidisciplinaryVisualizationProps {
  data: MultidisciplinaryData;
  title?: string;
  subtitle?: string;
  height?: number;
  width?: string;
  showLabels?: boolean;
  showLegend?: boolean;
  interactive?: boolean;
  className?: string;
  loading?: boolean;
  error?: string;
  onNodeClick?: (node: Node) => void;
  onNodeHover?: (node: Node | null) => void;
  onLinkClick?: (link: Link) => void;
}

const NODE_COLORS = {
  pattern: '#3B82F6', // blue
  cultural: '#10B981', // green
  environmental: '#F59E0B', // amber
  validation: '#EF4444', // red
  agent: '#8B5CF6' // purple
};

const LINK_COLORS = {
  cultural: '#10B981',
  environmental: '#F59E0B',
  validation: '#EF4444',
  agent_interaction: '#8B5CF6'
};

export function MultidisciplinaryVisualization({
  data,
  title,
  subtitle,
  height = 600,
  width = '100%',
  showLabels = true,
  showLegend = true,
  interactive = true,
  className,
  loading = false,
  error,
  onNodeClick,
  onNodeHover,
  onLinkClick
}: MultidisciplinaryVisualizationProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [hoveredNode, setHoveredNode] = useState<Node | null>(null);

  // Process data for D3 force simulation
  const processedData = useMemo(() => {
    if (!data) return { nodes: [], links: [] };

    // Create node map for link resolution
    const nodeMap = new Map(data.nodes.map(node => [node.id, node]));

    // Process links to use node objects instead of IDs
    const processedLinks = data.links.map(link => ({
      ...link,
      source: nodeMap.get(link.source)!,
      target: nodeMap.get(link.target)!
    }));

    return {
      nodes: data.nodes,
      links: processedLinks
    };
  }, [data]);

  useEffect(() => {
    if (!svgRef.current || !processedData.nodes.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove(); // Clear previous content

    const containerRect = svgRef.current.getBoundingClientRect();
    const width = containerRect.width;
    const height = containerRect.height;

    // Create main group
    const g = svg.append('g').attr('class', 'main-group');

    // Create zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom);

    // Create force simulation
    const simulation = d3.forceSimulation(processedData.nodes as d3.SimulationNodeDatum[])
      .force('link', d3.forceLink(processedData.links)
        .id((d: any) => d.id)
        .distance((d: any) => 100 / d.strength)
        .strength((d: any) => d.strength * 0.1)
      )
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius((d: any) => Math.sqrt(d.size) * 2 + 10));

    // Create links
    const link = g.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(processedData.links)
      .enter().append('line')
      .attr('stroke', (d: any) => LINK_COLORS[d.type] || '#999')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', (d: any) => Math.sqrt(d.strength) * 2)
      .style('cursor', interactive ? 'pointer' : 'default');

    // Create link labels
    const linkLabels = g.append('g')
      .attr('class', 'link-labels')
      .selectAll('text')
      .data(processedData.links)
      .enter().append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', -5)
      .attr('font-size', '10px')
      .attr('fill', '#666')
      .style('pointer-events', 'none')
      .style('opacity', 0)
      .text((d: any) => d.type);

    // Create nodes
    const node = g.append('g')
      .attr('class', 'nodes')
      .selectAll('circle')
      .data(processedData.nodes)
      .enter().append('circle')
      .attr('r', (d: any) => Math.sqrt(d.size) * 2)
      .attr('fill', (d: any) => NODE_COLORS[d.type] || '#999')
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', interactive ? 'pointer' : 'default');

    // Create node labels
    const nodeLabels = g.append('g')
      .attr('class', 'node-labels')
      .selectAll('text')
      .data(processedData.nodes)
      .enter().append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '.35em')
      .attr('font-size', '12px')
      .attr('fill', '#333')
      .style('pointer-events', 'none')
      .style('opacity', showLabels ? 1 : 0)
      .text((d: any) => d.label.length > 15 ? d.label.substring(0, 12) + '...' : d.label);

    // Add interactivity
    if (interactive) {
      node
        .on('mouseover', function(event, d: any) {
          // Highlight connected nodes and links
          const connectedNodeIds = new Set();
          processedData.links.forEach(link => {
            if (link.source.id === d.id) connectedNodeIds.add(link.target.id);
            if (link.target.id === d.id) connectedNodeIds.add(link.source.id);
          });

          node.attr('opacity', (n: any) =>
            n.id === d.id || connectedNodeIds.has(n.id) ? 1 : 0.3
          );
          link.attr('opacity', (l: any) =>
            l.source.id === d.id || l.target.id === d.id ? 1 : 0.1
          );

          setHoveredNode(d);
          if (onNodeHover) onNodeHover(d);
        })
        .on('mouseout', function() {
          node.attr('opacity', 1);
          link.attr('opacity', 0.6);

          setHoveredNode(null);
          if (onNodeHover) onNodeHover(null);
        })
        .on('click', function(event, d: any) {
          event.stopPropagation();
          setSelectedNode(d);
          if (onNodeClick) onNodeClick(d);
        });

      link
        .on('mouseover', function(event, d: any) {
          linkLabels.filter((l: any) => l === d).style('opacity', 1);
        })
        .on('mouseout', function(event, d: any) {
          linkLabels.filter((l: any) => l === d).style('opacity', 0);
        })
        .on('click', function(event, d: any) {
          if (onLinkClick) onLinkClick(d);
        });

      // Clear selection on background click
      svg.on('click', () => {
        setSelectedNode(null);
      });
    }

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node
        .attr('cx', (d: any) => d.x)
        .attr('cy', (d: any) => d.y);

      nodeLabels
        .attr('x', (d: any) => d.x)
        .attr('y', (d: any) => d.y);

      linkLabels
        .attr('x', (d: any) => (d.source.x + d.target.x) / 2)
        .attr('y', (d: any) => (d.source.y + d.target.y) / 2);
    });

    // Cleanup
    return () => {
      simulation.stop();
    };
  }, [processedData, showLabels, interactive, onNodeClick, onNodeHover, onLinkClick]);

  if (loading) {
    return (
      <div className={clsx('flex items-center justify-center border rounded-lg', className)} style={{ height, width }}>
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-2 text-sm text-gray-500">Loading multidisciplinary data...</p>
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

  if (!data || !data.nodes.length) {
    return (
      <div className={clsx('flex items-center justify-center border rounded-lg', className)} style={{ height, width }}>
        <div className="text-center">
          <div className="text-gray-400 mb-2">
            <svg className="h-8 w-8 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
            </svg>
          </div>
          <p className="text-sm text-gray-600">No multidisciplinary data to display</p>
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

      <div className="relative border rounded-lg overflow-hidden bg-white" style={{ height, width }}>
        <svg
          ref={svgRef}
          width="100%"
          height="100%"
          className="bg-gray-50"
        />

        {/* Selected node details */}
        {selectedNode && (
          <div className="absolute top-4 left-4 bg-white p-4 rounded-lg shadow-lg z-10 max-w-sm">
            <h4 className="font-medium text-gray-900 mb-2">{selectedNode.label}</h4>
            <div className="space-y-1 text-sm text-gray-600">
              <p><span className="font-medium">Type:</span> {selectedNode.type}</p>
              <p><span className="font-medium">Group:</span> {selectedNode.group}</p>
              {selectedNode.properties.confidence && (
                <p><span className="font-medium">Confidence:</span> {(selectedNode.properties.confidence * 100).toFixed(1)}%</p>
              )}
              {selectedNode.properties.relevance && (
                <p><span className="font-medium">Relevance:</span> {(selectedNode.properties.relevance * 100).toFixed(1)}%</p>
              )}
              {selectedNode.properties.connections && (
                <p><span className="font-medium">Connections:</span> {selectedNode.properties.connections}</p>
              )}
            </div>
          </div>
        )}

        {/* Legend */}
        {showLegend && (
          <div className="absolute bottom-4 right-4 bg-white p-3 rounded-lg shadow-lg z-10">
            <h4 className="text-sm font-medium text-gray-900 mb-2">Node Types</h4>
            <div className="space-y-1">
              {Object.entries(NODE_COLORS).map(([type, color]) => (
                <div key={type} className="flex items-center space-x-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: color }}
                  />
                  <span className="text-xs text-gray-600 capitalize">{type}</span>
                </div>
              ))}
            </div>
            <h4 className="text-sm font-medium text-gray-900 mt-3 mb-2">Link Types</h4>
            <div className="space-y-1">
              {Object.entries(LINK_COLORS).map(([type, color]) => (
                <div key={type} className="flex items-center space-x-2">
                  <div
                    className="w-4 h-0.5"
                    style={{ backgroundColor: color }}
                  />
                  <span className="text-xs text-gray-600 capitalize">{type.replace('_', ' ')}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default MultidisciplinaryVisualization;