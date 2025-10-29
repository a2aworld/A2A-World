/**
 * A2A World Platform - Validation Components
 * 
 * Export all validation dashboard components for the statistical validation framework.
 */

export { StatisticalValidation } from './StatisticalValidation';
export type { 
  StatisticalValidationResult,
  MoransIResult,
  NullHypothesisResult,
  SpatialStatisticsResult,
  SignificanceClassification,
  StatisticalValidationProps
} from './StatisticalValidation';

export { SignificanceIndicators } from './SignificanceIndicators';
export type { SignificanceIndicatorsProps } from './SignificanceIndicators';

export { ValidationDashboard } from './ValidationDashboard';
export type { ValidationDashboardProps, DashboardData } from './ValidationDashboard';

export { ValidationReports } from './ValidationReports';
export type { 
  ValidationReport,
  ValidationReportsProps
} from './ValidationReports';

// Default export for convenience
export { ValidationDashboard as default } from './ValidationDashboard';