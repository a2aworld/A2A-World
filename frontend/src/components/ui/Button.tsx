/**
 * A2A World Platform - Button Component
 * 
 * Reusable button component with different variants and states.
 */

import { clsx } from 'clsx';
import { LucideIcon } from 'lucide-react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  icon?: LucideIcon;
  iconPosition?: 'left' | 'right';
}

export function Button({
  children,
  className,
  variant = 'primary',
  size = 'md',
  loading = false,
  icon: Icon,
  iconPosition = 'left',
  disabled,
  ...props
}: ButtonProps) {
  const buttonClasses = clsx(
    'inline-flex items-center justify-center font-medium rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed',
    {
      // Variants
      'bg-primary-600 hover:bg-primary-700 text-white focus:ring-primary-500': variant === 'primary',
      'bg-gray-200 hover:bg-gray-300 text-gray-800 focus:ring-gray-500': variant === 'secondary',
      'border border-gray-300 bg-white hover:bg-gray-50 text-gray-700 focus:ring-primary-500': variant === 'outline',
      'hover:bg-gray-100 text-gray-700 focus:ring-primary-500': variant === 'ghost',
      'bg-red-600 hover:bg-red-700 text-white focus:ring-red-500': variant === 'danger',
      
      // Sizes
      'px-3 py-2 text-sm': size === 'sm',
      'px-4 py-2 text-sm': size === 'md',
      'px-6 py-3 text-base': size === 'lg',
    },
    className
  );

  const iconClasses = clsx(
    {
      'mr-2': Icon && iconPosition === 'left' && children,
      'ml-2': Icon && iconPosition === 'right' && children,
      'animate-spin': loading,
    }
  );

  return (
    <button
      className={buttonClasses}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? (
        <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
      ) : Icon && iconPosition === 'left' ? (
        <Icon className={clsx('h-4 w-4', iconClasses)} />
      ) : null}
      
      {children}
      
      {!loading && Icon && iconPosition === 'right' && (
        <Icon className={clsx('h-4 w-4', iconClasses)} />
      )}
    </button>
  );
}

export default Button;