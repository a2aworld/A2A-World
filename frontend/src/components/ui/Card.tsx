/**
 * A2A World Platform - Card Component
 * 
 * Reusable card component for displaying content with consistent styling.
 */

import { clsx } from 'clsx';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  padding?: 'sm' | 'md' | 'lg';
  variant?: 'default' | 'bordered' | 'elevated';
  header?: React.ReactNode;
  footer?: React.ReactNode;
}

export function Card({ 
  children, 
  className, 
  padding = 'md', 
  variant = 'default',
  header,
  footer 
}: CardProps) {
  const cardClasses = clsx(
    'bg-white rounded-lg',
    {
      'shadow-sm border border-gray-200': variant === 'bordered',
      'shadow-md': variant === 'default',
      'shadow-lg': variant === 'elevated',
      'p-3': padding === 'sm',
      'p-6': padding === 'md',
      'p-8': padding === 'lg',
    },
    className
  );

  return (
    <div className={cardClasses}>
      {header && (
        <div className="border-b border-gray-200 pb-4 mb-4">
          {header}
        </div>
      )}
      {children}
      {footer && (
        <div className="border-t border-gray-200 pt-4 mt-4">
          {footer}
        </div>
      )}
    </div>
  );
}

interface CardHeaderProps {
  title: string;
  subtitle?: string;
  action?: React.ReactNode;
  className?: string;
}

export function CardHeader({ title, subtitle, action, className }: CardHeaderProps) {
  return (
    <div className={clsx('flex justify-between items-start', className)}>
      <div>
        <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
        {subtitle && (
          <p className="text-sm text-gray-600 mt-1">{subtitle}</p>
        )}
      </div>
      {action && <div>{action}</div>}
    </div>
  );
}

export default Card;