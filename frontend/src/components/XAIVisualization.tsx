/**
 * A2A World Platform - Narrative-Driven XAI Visualization Component
 *
 * Explainable AI visualization showing decision-making processes, confidence levels,
 * and narrative explanations for pattern discovery and validation results.
 */

import React, { useRef, useEffect, useState, useMemo } from 'react';
import * as d3 from 'd3';
import { clsx } from 'clsx';

export interface DecisionNode {
  id: string;
  label: string;
  type: 'input' | 'process' | 'decision' | 'output' | 'evidence';
  confidence: number;
  explanation: string;
  evidence: string[];
  x?: number;
  y?: number;
  children?: string[];
  parent?: string;
}

export interface DecisionLink {
  source: string;
  target: string;
  type: 'flow' | 'evidence' | 'alternative';
  label?: string;
  strength: number;
}

export interface XAIDecisionTree {
  nodes: DecisionNode[];
  links: DecisionLink[];
  rootId: string;
  conclusion: string;
  overallConfidence: number;
}

export interface XAIVisualizationProps {
  data: XAIDecisionTree;
  title?: string;
  subtitle?: string;
  height?: number;
  width?: string;
  showExplanations?: boolean;
  showConfidence?: boolean;
  interactive?: boolean;
  layout?: 'tree' | 'radial' | 'force';
  className?: string;
  loading?: boolean;
  error?: string;
  onNodeClick?: (node: DecisionNode) => void;
  onNodeHover?: (node: DecisionNode | null) => void;
}

const NODE_COLORS = {
  input: '#3B82F6', // blue
  process: '#F59E0B', // amber
  decision: '#EF4444', // red
  output: '#10B981', // green
  evidence: '#8B5CF6' // purple
};

const LINK_COLORS = {
  flow: '#6B7280',
  evidence: '#8B5CF6',
  alternative: '#EF4444'
};

export function XAIVisualization({
  data,
  title,
  subtitle,
  height = 600,
  width = '100%',
  showExplanations = true,
  showConfidence = true,
  interactive = true,
  layout = 'tree',
  className,
  loading = false,
  error,
  onNodeClick,
  onNodeHover
}: XAIVisualizationProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedNode, setSelectedNode] = useState<DecisionNode | null>(null);
  const [hoveredNode, setHoveredNode] = useState<DecisionNode | null>(null);

  // Process data for visualization
  const processedData = useMemo(() => {
    if (!data) return { nodes: [], links: [] };

    // Create node map for link resolution
    const nodeMap = new Map(data.nodes.map(node => [node.id, node]));

    // Process links to use node objects
    const processedLinks = data.links.map(link => ({
      ...link,
      source: nodeMap.get(link.source)!,
      target: nodeMap.get(link.target)!
    }));

    return {
      nodes: data.nodes,
      links: processedLinks,
      root: nodeMap.get(data.rootId)
    };
  }, [data]);

  useEffect(() => {
    if (!svgRef.current || !processedData.nodes.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const containerRect = svgRef.current.getBoundingClientRect();
    const width = containerRect.width;
    const height = containerRect.height;

    // Create main group
    const g = svg.append('g').attr('class', 'main-group');

    // Create zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 2])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom);

    let simulation: d3.Simulation<d3.SimulationNodeDatum, undefined>;

    if (layout === 'tree') {
      // Tree layout
      const treeLayout = d3.tree<DecisionNode>()
        .size([width - 100, height - 100])
        .nodeSize([120, 200]);

      const root = d3.hierarchy(processedData.root!, (d) => {
        return data.nodes.filter(n => d.children?.includes(n.id));
      });

      const treeData = treeLayout(root);

      // Create links
      const link = g.append('g')
        .attr('class', 'links')
        .selectAll('path')
        .data(treeData.links())
        .enter().append('path')
        .attr('d', d3.linkHorizontal()
          .x((d: any) => d.y + 50)
          .y((d: any) => d.x + height / 2)
        )
        .attr('stroke', (d: any) => LINK_COLORS.flow)
        .attr('stroke-width', 2)
        .attr('fill', 'none')
        .attr('marker-end', 'url(#arrowhead)');

      // Create nodes
      const node = g.append('g')
        .attr('class', 'nodes')
        .selectAll('g')
        .data(treeData.descendants())
        .enter().append('g')
        .attr('transform', (d: any) => `translate(${d.y + 50},${d.x + height / 2})`);

      // Node circles
      node.append('circle')
        .attr('r', (d: any) => Math.max(20, Math.sqrt(d.data.confidence) * 30))
        .attr('fill', (d: any) => NODE_COLORS[d.data.type] || '#999')
        .attr('stroke', '#fff')
        .attr('stroke-width', 3)
        .style('cursor', interactive ? 'pointer' : 'default');

      // Node labels
      node.append('text')
        .attr('dy', -35)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('font-weight', 'bold')
        .attr('fill', '#333')
        .text((d: any) => d.data.label.length > 20 ? d.data.label.substring(0, 17) + '...' : d.data.label);

      // Confidence indicators
      if (showConfidence) {
        node.append('text')
          .attr('dy', 45)
          .attr('text-anchor', 'middle')
          .attr('font-size', '10px')
          .attr('fill', '#666')
          .text((d: any) => `${(d.data.confidence * 100).toFixed(0)}%`);
      }

      // Add interactivity
      if (interactive) {
        node
          .on('mouseover', function(event, d: any) {
            setHoveredNode(d.data);
            if (onNodeHover) onNodeHover(d.data);
          })
          .on('mouseout', function() {
            setHoveredNode(null);
            if (onNodeHover) onNodeHover(null);
          })
          .on('click', function(event, d: any) {
            event.stopPropagation();
            setSelectedNode(d.data);
            if (onNodeClick) onNodeClick(d.data);
          });
      }

    } else if (layout === 'force') {
      // Force-directed layout
      simulation = d3.forceSimulation(processedData.nodes as d3.SimulationNodeDatum[])
        .force('link', d3.forceLink(processedData.links)
          .id((d: any) => d.id)
          .distance(150)
          .strength(0.5)
        )
        .force('charge', d3.forceManyBody().strength(-500))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius((d: any) => Math.max(25, Math.sqrt(d.confidence) * 35)));

      // Create links
      const link = g.append('g')
        .attr('class', 'links')
        .selectAll('line')
        .data(processedData.links)
        .enter().append('line')
        .attr('stroke', (d: any) => LINK_COLORS[d.type] || '#999')
        .attr('stroke-width', (d: any) => d.strength * 3)
        .attr('stroke-dasharray', (d: any) => d.type === 'alternative' ? '5,5' : 'none')
        .attr('marker-end', 'url(#arrowhead)');

      // Create nodes
      const node = g.append('g')
        .attr('class', 'nodes')
        .selectAll('g')
        .data(processedData.nodes)
        .enter().append('g');

      // Node circles
      node.append('circle')
        .attr('r', (d: any) => Math.max(20, Math.sqrt(d.confidence) * 30))
        .attr('fill', (d: any) => NODE_COLORS[d.type] || '#999')
        .attr('stroke', '#fff')
        .attr('stroke-width', 3)
        .style('cursor', interactive ? 'pointer' : 'default');

      // Node labels
      node.append('text')
        .attr('dy', -35)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('font-weight', 'bold')
        .attr('fill', '#333')
        .text((d: any) => d.label.length > 15 ? d.label.substring(0, 12) + '...' : d.label);

      // Confidence indicators
      if (showConfidence) {
        node.append('text')
          .attr('dy', 45)
          .attr('text-anchor', 'middle')
          .attr('font-size', '10px')
          .attr('fill', '#666')
          .text((d: any) => `${(d.confidence * 100).toFixed(0)}%`);
      }

      // Add interactivity
      if (interactive) {
        node
          .on('mouseover', function(event, d: any) {
            setHoveredNode(d);
            if (onNodeHover) onNodeHover(d);
          })
          .on('mouseout', function() {
            setHoveredNode(null);
            if (onNodeHover) onNodeHover(null);
          })
          .on('click', function(event, d: any) {
            event.stopPropagation();
            setSelectedNode(d);
            if (onNodeClick) onNodeClick(d);
          });
      }

      // Update positions on simulation tick
      simulation.on('tick', () => {
        link
          .attr('x1', (d: any) => d.source.x)
          .attr('y1', (d: any) => d.source.y)
          .attr('x2', (d: any) => d.target.x)
          .attr('y2', (d: any) => d.target.y);

        node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
      });
    }

    // Add arrow marker
    svg.append('defs').append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 20)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#666');

    // Cleanup
    return () => {
      if (simulation) {
        simulation.stop();
      }
    };
  }, [processedData, layout, showConfidence, interactive, onNodeClick, onNodeHover]);

  if (loading) {
    return (
      <div className={clsx('flex items-center justify-center border rounded-lg', className)} style={{ height, width }}>
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-2 text-sm text-gray-500">Loading XAI explanation...</p>
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
          <p className="text-sm text-gray-600">No XAI data to display</p>
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
          <div className="mt-2 flex items-center space-x-4">
            <div className="text-sm text-gray-600">
              <span className="font-medium">Overall Confidence:</span> {(data.overallConfidence * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-600">
              <span className="font-medium">Conclusion:</span> {data.conclusion}
            </div>
          </div>
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
        {selectedNode && showExplanations && (
          <div className="absolute top-4 left-4 bg-white p-4 rounded-lg shadow-lg z-10 max-w-md">
            <h4 className="font-medium text-gray-900 mb-2">{selectedNode.label}</h4>
            <div className="space-y-2 text-sm">
              <p><span className="font-medium">Type:</span> {selectedNode.type}</p>
              <p><span className="font-medium">Confidence:</span> {(selectedNode.confidence * 100).toFixed(1)}%</p>
              <div>
                <span className="font-medium">Explanation:</span>
                <p className="mt-1 text-gray-600">{selectedNode.explanation}</p>
              </div>
              {selectedNode.evidence.length > 0 && (
                <div>
                  <span className="font-medium">Evidence:</span>
                  <ul className="mt-1 ml-4 list-disc text-gray-600">
                    {selectedNode.evidence.map((evidence, index) => (
                      <li key={index}>{evidence}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Legend */}
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
      </div>
    </div>
  );
}

export default XAIVisualization;