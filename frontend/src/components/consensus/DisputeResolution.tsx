/**
 * A2A World Platform - Dispute Resolution Component
 * 
 * Interface for handling validation disputes when consensus cannot be reached
 * or when there are significant disagreements between agents or statistical evidence.
 */

import React, { useState, useEffect } from 'react';
import { 
  AlertTriangle, 
  Scale, 
  MessageSquare, 
  User, 
  Clock,
  CheckCircle,
  XCircle,
  FileText,
  Send,
  Eye,
  ThumbsUp,
  ThumbsDown
} from 'lucide-react';
import { BrandedCard } from '../branding/BrandedCard';
import { LoadingSpinner } from '../branding/LoadingSpinner';

// Types for dispute resolution
interface DisputeCase {
  id: string;
  pattern_id: string;
  dispute_type: 'consensus_failure' | 'statistical_conflict' | 'agent_disagreement';
  status: 'open' | 'reviewing' | 'resolved' | 'escalated';
  severity: 'low' | 'medium' | 'high' | 'critical';
  created_by: string;
  assigned_reviewers: string[];
  consensus_data: {
    votes: Array<{
      agent_id: string;
      vote: string;
      confidence: number;
      reasoning: string;
    }>;
    statistical_evidence: Record<string, any>;
    conflict_points: string[];
  };
  resolution_data?: {
    decision: string;
    reasoning: string;
    decided_by: string;
    decision_timestamp: string;
  };
  comments: Array<{
    id: string;
    author: string;
    content: string;
    timestamp: string;
    type: 'comment' | 'evidence' | 'recommendation';
  }>;
  created_at: string;
  updated_at: string;
}

interface DisputeResolutionProps {
  disputeId?: string;
  patternId?: string;
  onResolutionComplete?: (resolution: any) => void;
}

export const DisputeResolution: React.FC<DisputeResolutionProps> = ({
  disputeId,
  patternId,
  onResolutionComplete
}) => {
  const [dispute, setDispute] = useState<DisputeCase | null>(null);
  const [disputes, setDisputes] = useState<DisputeCase[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [newComment, setNewComment] = useState('');
  const [selectedView, setSelectedView] = useState<'overview' | 'evidence' | 'discussion'>('overview');

  useEffect(() => {
    if (disputeId) {
      fetchDispute(disputeId);
    } else if (patternId) {
      fetchDisputesForPattern(patternId);
    } else {
      fetchRecentDisputes();
    }
  }, [disputeId, patternId]);

  const fetchDispute = async (id: string) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Simulate API call - in real implementation would fetch from API
      const simulatedDispute = generateSimulatedDispute(id);
      setDispute(simulatedDispute);
    } catch (err) {
      console.error('Error fetching dispute:', err);
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const fetchDisputesForPattern = async (pattern_id: string) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Simulate pattern-specific disputes
      const simulatedDisputes = [generateSimulatedDispute('1', pattern_id)];
      setDisputes(simulatedDisputes);
    } catch (err) {
      console.error('Error fetching disputes for pattern:', err);
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const fetchRecentDisputes = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Simulate recent disputes
      const simulatedDisputes = Array.from({ length: 5 }, (_, i) => 
        generateSimulatedDispute((i + 1).toString())
      );
      setDisputes(simulatedDisputes);
    } catch (err) {
      console.error('Error fetching recent disputes:', err);
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const generateSimulatedDispute = (id: string, pattern_id?: string): DisputeCase => {
    const disputeTypes = ['consensus_failure', 'statistical_conflict', 'agent_disagreement'] as const;
    const statuses = ['open', 'reviewing', 'resolved'] as const;
    const severities = ['low', 'medium', 'high', 'critical'] as const;
    
    return {
      id: `dispute_${id}`,
      pattern_id: pattern_id || `pattern_${Math.random().toString(36).substr(2, 9)}`,
      dispute_type: disputeTypes[Math.floor(Math.random() * disputeTypes.length)],
      status: statuses[Math.floor(Math.random() * statuses.length)],
      severity: severities[Math.floor(Math.random() * severities.length)],
      created_by: `agent_${Math.floor(Math.random() * 5) + 1}`,
      assigned_reviewers: [`reviewer_${Math.floor(Math.random() * 3) + 1}`],
      consensus_data: {
        votes: [
          {
            agent_id: 'agent_1',
            vote: 'significant',
            confidence: 0.85,
            reasoning: 'Strong statistical evidence supports pattern significance'
          },
          {
            agent_id: 'agent_2',
            vote: 'not_significant',
            confidence: 0.78,
            reasoning: 'Statistical analysis shows insufficient evidence'
          },
          {
            agent_id: 'agent_3',
            vote: 'uncertain',
            confidence: 0.45,
            reasoning: 'Mixed statistical results require further analysis'
          }
        ],
        statistical_evidence: {
          morans_i: 0.45,
          p_value: 0.08,
          sample_size: 25
        },
        conflict_points: [
          'Disagreement on statistical significance threshold',
          'Different interpretations of Moran\'s I result',
          'Sample size adequacy concerns'
        ]
      },
      comments: [
        {
          id: 'comment_1',
          author: 'reviewer_1',
          content: 'Need to review the statistical methodology used by different agents',
          timestamp: new Date(Date.now() - 3600000).toISOString(),
          type: 'comment'
        },
        {
          id: 'comment_2',
          author: 'agent_1',
          content: 'Additional bootstrap validation confirms significance',
          timestamp: new Date(Date.now() - 1800000).toISOString(),
          type: 'evidence'
        }
      ],
      created_at: new Date(Date.now() - 7200000).toISOString(),
      updated_at: new Date().toISOString()
    };
  };

  const handleAddComment = async () => {
    if (!newComment.trim() || !dispute) return;
    
    try {
      const comment = {
        id: `comment_${Date.now()}`,
        author: 'current_user', // Would come from auth context
        content: newComment.trim(),
        timestamp: new Date().toISOString(),
        type: 'comment' as const
      };
      
      // Update dispute with new comment
      const updatedDispute = {
        ...dispute,
        comments: [...dispute.comments, comment],
        updated_at: new Date().toISOString()
      };
      
      setDispute(updatedDispute);
      setNewComment('');
      
      // In real implementation, would send to API
      console.log('Added comment to dispute:', comment);
      
    } catch (err) {
      console.error('Error adding comment:', err);
    }
  };

  const handleResolveDispute = async (decision: string, reasoning: string) => {
    if (!dispute) return;
    
    try {
      const resolution = {
        decision,
        reasoning,
        decided_by: 'current_user',
        decision_timestamp: new Date().toISOString()
      };
      
      const resolvedDispute = {
        ...dispute,
        status: 'resolved' as const,
        resolution_data: resolution,
        updated_at: new Date().toISOString()
      };
      
      setDispute(resolvedDispute);
      
      if (onResolutionComplete) {
        onResolutionComplete(resolution);
      }
      
      console.log('Resolved dispute:', resolution);
      
    } catch (err) {
      console.error('Error resolving dispute:', err);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'text-red-600 bg-red-100';
      case 'high':
        return 'text-orange-600 bg-orange-100';
      case 'medium':
        return 'text-yellow-600 bg-yellow-100';
      case 'low':
        return 'text-green-600 bg-green-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'resolved':
        return 'text-green-600 bg-green-100';
      case 'reviewing':
        return 'text-blue-600 bg-blue-100';
      case 'escalated':
        return 'text-red-600 bg-red-100';
      case 'open':
        return 'text-yellow-600 bg-yellow-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  if (isLoading) {
    return (
      <BrandedCard className="p-6">
        <div className="flex items-center justify-center">
          <LoadingSpinner size="lg" />
          <span className="ml-3 text-lg">Loading dispute information...</span>
        </div>
      </BrandedCard>
    );
  }

  if (error) {
    return (
      <BrandedCard className="p-6 border-red-200 bg-red-50">
        <div className="flex items-center mb-4">
          <AlertTriangle className="h-6 w-6 text-red-500 mr-3" />
          <h3 className="text-lg font-semibold text-red-700">Dispute Resolution Error</h3>
        </div>
        <p className="text-red-600">{error}</p>
      </BrandedCard>
    );
  }

  // Single dispute view
  if (dispute) {
    return (
      <div className="space-y-6">
        {/* Dispute Header */}
        <BrandedCard className="p-6">
          <div className="flex items-start justify-between mb-6">
            <div className="flex items-center">
              <Scale className="h-8 w-8 text-blue-600 mr-4" />
              <div>
                <h2 className="text-xl font-bold text-gray-900">
                  Dispute Resolution
                </h2>
                <p className="text-sm text-gray-600">
                  Dispute ID: {dispute.id} | Pattern: {dispute.pattern_id}
                </p>
              </div>
            </div>
            
            <div className="flex flex-col items-end space-y-2">
              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getSeverityColor(dispute.severity)}`}>
                {dispute.severity.toUpperCase()}
              </span>
              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(dispute.status)}`}>
                {dispute.status.replace('_', ' ').toUpperCase()}
              </span>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex space-x-1 mb-6">
            {[
              { id: 'overview', label: 'Overview', icon: FileText },
              { id: 'evidence', label: 'Evidence', icon: Eye },
              { id: 'discussion', label: 'Discussion', icon: MessageSquare }
            ].map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setSelectedView(id as any)}
                className={`flex items-center px-4 py-2 text-sm font-medium rounded-lg ${
                  selectedView === id
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                <Icon className="h-4 w-4 mr-2" />
                {label}
              </button>
            ))}
          </div>

          {/* Dispute Overview */}
          {selectedView === 'overview' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-lg font-semibold text-gray-900 mb-3">Dispute Details</h4>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Type:</span>
                      <span className="font-medium">{dispute.dispute_type.replace('_', ' ')}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Created:</span>
                      <span className="font-medium">{new Date(dispute.created_at).toLocaleDateString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Created by:</span>
                      <span className="font-medium">{dispute.created_by}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Reviewers:</span>
                      <span className="font-medium">{dispute.assigned_reviewers.join(', ')}</span>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h4 className="text-lg font-semibold text-gray-900 mb-3">Voting Summary</h4>
                  <div className="space-y-2">
                    {dispute.consensus_data.votes.map((vote, index) => (
                      <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                        <span className="font-medium">{vote.agent_id}</span>
                        <div className="flex items-center space-x-2">
                          <span className={`text-sm ${
                            vote.vote === 'significant' ? 'text-green-600' :
                            vote.vote === 'not_significant' ? 'text-red-600' : 'text-yellow-600'
                          }`}>
                            {vote.vote}
                          </span>
                          <span className="text-sm text-gray-500">
                            ({(vote.confidence * 100).toFixed(0)}%)
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Conflict Points */}
              <div>
                <h4 className="text-lg font-semibold text-gray-900 mb-3">Key Conflict Points</h4>
                <ul className="space-y-2">
                  {dispute.consensus_data.conflict_points.map((point, index) => (
                    <li key={index} className="flex items-start">
                      <AlertTriangle className="h-4 w-4 text-yellow-500 mr-2 mt-0.5 flex-shrink-0" />
                      <span className="text-gray-700">{point}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Statistical Evidence */}
          {selectedView === 'evidence' && (
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-gray-900">Statistical Evidence</h4>
              
              <div className="bg-blue-50 p-4 rounded-lg">
                <h5 className="font-medium text-blue-800 mb-2">Primary Statistics</h5>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {Object.entries(dispute.consensus_data.statistical_evidence).map(([key, value]) => (
                    <div key={key} className="text-center">
                      <div className="text-sm text-blue-600 font-medium">
                        {key.replace('_', ' ').toUpperCase()}
                      </div>
                      <div className="text-lg font-bold text-blue-800">
                        {typeof value === 'number' ? value.toFixed(3) : String(value)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Agent Reasoning */}
              <div className="space-y-3">
                <h5 className="font-medium text-gray-900">Agent Reasoning</h5>
                {dispute.consensus_data.votes.map((vote, index) => (
                  <div key={index} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium text-gray-900">{vote.agent_id}</span>
                      <div className="flex items-center space-x-2">
                        <span className={`text-sm font-medium ${
                          vote.vote === 'significant' ? 'text-green-600' :
                          vote.vote === 'not_significant' ? 'text-red-600' : 'text-yellow-600'
                        }`}>
                          {vote.vote.replace('_', ' ')}
                        </span>
                        <span className="text-sm text-gray-500">
                          ({(vote.confidence * 100).toFixed(0)}% confidence)
                        </span>
                      </div>
                    </div>
                    <p className="text-gray-700 text-sm">{vote.reasoning}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Discussion */}
          {selectedView === 'discussion' && (
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-gray-900">Discussion</h4>
              
              {/* Comments */}
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {dispute.comments.map((comment) => (
                  <div key={comment.id} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center">
                        <User className="h-4 w-4 text-gray-500 mr-2" />
                        <span className="font-medium text-gray-900">{comment.author}</span>
                        <span className={`ml-2 px-2 py-0.5 rounded text-xs ${
                          comment.type === 'evidence' ? 'bg-blue-100 text-blue-800' :
                          comment.type === 'recommendation' ? 'bg-green-100 text-green-800' :
                          'bg-gray-100 text-gray-800'
                        }`}>
                          {comment.type}
                        </span>
                      </div>
                      <span className="text-sm text-gray-500">
                        {new Date(comment.timestamp).toLocaleString()}
                      </span>
                    </div>
                    <p className="text-gray-700">{comment.content}</p>
                  </div>
                ))}
              </div>

              {/* Add Comment */}
              {dispute.status !== 'resolved' && (
                <div className="border-t border-gray-200 pt-4">
                  <div className="flex space-x-3">
                    <div className="flex-1">
                      <textarea
                        value={newComment}
                        onChange={(e) => setNewComment(e.target.value)}
                        placeholder="Add your comment or evidence..."
                        rows={3}
                        className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                      />
                    </div>
                    <button
                      onClick={handleAddComment}
                      disabled={!newComment.trim()}
                      className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
                    >
                      <Send className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Resolution Actions */}
          {dispute.status !== 'resolved' && selectedView === 'overview' && (
            <div className="border-t border-gray-200 pt-6 mt-6">
              <h4 className="text-lg font-semibold text-gray-900 mb-4">Resolution Actions</h4>
              
              <div className="flex flex-wrap gap-3">
                <button
                  onClick={() => handleResolveDispute('significant', 'Resolved as significant based on consensus review')}
                  className="flex items-center px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
                >
                  <ThumbsUp className="h-4 w-4 mr-2" />
                  Resolve as Significant
                </button>
                
                <button
                  onClick={() => handleResolveDispute('not_significant', 'Resolved as not significant based on consensus review')}
                  className="flex items-center px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
                >
                  <ThumbsDown className="h-4 w-4 mr-2" />
                  Resolve as Not Significant
                </button>
                
                <button
                  onClick={() => handleResolveDispute('escalate', 'Escalated for expert human review')}
                  className="flex items-center px-4 py-2 bg-orange-600 text-white rounded-md hover:bg-orange-700"
                >
                  <AlertTriangle className="h-4 w-4 mr-2" />
                  Escalate to Expert
                </button>
              </div>
            </div>
          )}
        </BrandedCard>
      </div>
    );
  }

  // Multiple disputes list view
  return (
    <div className="space-y-6">
      <BrandedCard className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center">
            <Scale className="h-6 w-6 text-blue-600 mr-3" />
            <h2 className="text-xl font-bold text-gray-900">
              Active Disputes
            </h2>
          </div>
          
          <span className="text-sm text-gray-600">
            {disputes.filter(d => d.status !== 'resolved').length} open disputes
          </span>
        </div>

        {disputes.length === 0 ? (
          <div className="text-center text-gray-500 py-8">
            No disputes found
          </div>
        ) : (
          <div className="space-y-4">
            {disputes.map((disputeItem) => (
              <div key={disputeItem.id} className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <h4 className="font-medium text-gray-900">
                        {disputeItem.dispute_type.replace('_', ' ').toUpperCase()}
                      </h4>
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${getSeverityColor(disputeItem.severity)}`}>
                        {disputeItem.severity}
                      </span>
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${getStatusColor(disputeItem.status)}`}>
                        {disputeItem.status}
                      </span>
                    </div>
                    
                    <p className="text-sm text-gray-600 mb-2">
                      Pattern: {disputeItem.pattern_id}
                    </p>
                    
                    <div className="flex items-center space-x-4 text-xs text-gray-500">
                      <span>Created: {new Date(disputeItem.created_at).toLocaleDateString()}</span>
                      <span>Votes: {disputeItem.consensus_data.votes.length}</span>
                      <span>Comments: {disputeItem.comments.length}</span>
                    </div>
                  </div>
                  
                  <button
                    onClick={() => setDispute(disputeItem)}
                    className="ml-4 px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
                  >
                    Review
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </BrandedCard>
    </div>
  );
};

export default DisputeResolution;