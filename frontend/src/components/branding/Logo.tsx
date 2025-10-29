/**
 * A2A World Platform - Logo Component
 * 
 * Reusable logo component with different variants, sizes, and styling options.
 * Supports the A2A Star Logo integration throughout the platform.
 */

import React from 'react';
import Image from 'next/image';
import { clsx } from 'clsx';

export interface LogoProps {
  /**
   * Logo variant to display
   */
  variant?: 'full' | 'icon' | 'text' | 'stacked';
  /**
   * Logo size
   */
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl' | 'custom';
  /**
   * Custom width (when size is 'custom')
   */
  width?: number;
  /**
   * Custom height (when size is 'custom')
   */
  height?: number;
  /**
   * Logo color theme
   */
  theme?: 'light' | 'dark' | 'auto';
  /**
   * Whether logo should be clickable (links to home)
   */
  clickable?: boolean;
  /**
   * Additional CSS classes
   */
  className?: string;
  /**
   * Whether to show text alongside logo
   */
  showText?: boolean;
  /**
   * Custom text to display (defaults to "A2A World")
   */
  text?: string;
  /**
   * Text position relative to logo
   */
  textPosition?: 'right' | 'below';
  /**
   * Loading state
   */
  loading?: boolean;
}

const sizeMap = {
  xs: { width: 20, height: 20 },
  sm: { width: 32, height: 32 },
  md: { width: 48, height: 48 },
  lg: { width: 64, height: 64 },
  xl: { width: 96, height: 96 }
};

const textSizeMap = {
  xs: 'text-sm',
  sm: 'text-base', 
  md: 'text-lg',
  lg: 'text-xl',
  xl: 'text-2xl'
};

export function Logo({
  variant = 'full',
  size = 'md',
  width: customWidth,
  height: customHeight,
  theme = 'auto',
  clickable = false,
  className,
  showText = true,
  text = 'A2A World',
  textPosition = 'right',
  loading = false
}: LogoProps) {
  // Determine dimensions
  const dimensions = size === 'custom' 
    ? { width: customWidth || 48, height: customHeight || 48 }
    : sizeMap[size];

  // Logo image element
  const logoImage = (
    <div className={clsx(
      'relative flex-shrink-0',
      loading && 'animate-pulse bg-gray-200 rounded'
    )}>
      {!loading ? (
        <Image
          src="/logo.png"
          alt="A2A World Logo"
          width={dimensions.width}
          height={dimensions.height}
          className={clsx(
            'object-contain',
            theme === 'dark' && 'invert',
            className
          )}
          priority
        />
      ) : (
        <div 
          className="bg-gray-200 rounded"
          style={{ 
            width: dimensions.width, 
            height: dimensions.height 
          }} 
        />
      )}
    </div>
  );

  // Text element (if needed)
  const logoText = showText && !loading && (variant === 'full' || variant === 'text' || variant === 'stacked') && (
    <div className={clsx(
      'flex flex-col',
      textPosition === 'right' && 'ml-3',
      textPosition === 'below' && 'mt-2 text-center'
    )}>
      <span className={clsx(
        'font-bold text-gray-900',
        theme === 'dark' && 'text-white',
        size === 'custom' ? 'text-lg' : textSizeMap[size]
      )}>
        {text}
      </span>
      {size !== 'xs' && size !== 'sm' && (
        <span className={clsx(
          'text-xs text-gray-600 -mt-1',
          theme === 'dark' && 'text-gray-300'
        )}>
          Pattern Discovery Platform
        </span>
      )}
    </div>
  );

  // Container based on variant
  const logoContainer = (
    <div className={clsx(
      'flex items-center',
      variant === 'stacked' && 'flex-col',
      textPosition === 'below' && variant !== 'stacked' && 'flex-col',
      className
    )}>
      {variant !== 'text' && logoImage}
      {logoText}
    </div>
  );

  // Return clickable or static version
  if (clickable && !loading) {
    return (
      <a 
        href="/"
        className={clsx(
          'flex items-center hover:opacity-80 transition-opacity',
          variant === 'stacked' && 'flex-col',
          textPosition === 'below' && variant !== 'stacked' && 'flex-col'
        )}
        aria-label="A2A World Home"
      >
        {variant !== 'text' && logoImage}
        {logoText}
      </a>
    );
  }

  return logoContainer;
}

/**
 * Specialized logo variants for common use cases
 */

export function HeaderLogo({ size = 'md', clickable = true }: Pick<LogoProps, 'size' | 'clickable'>) {
  return (
    <Logo
      variant="full"
      size={size}
      clickable={clickable}
      textPosition="right"
      showText={true}
    />
  );
}

export function FooterLogo({ size = 'sm', theme = 'dark' }: Pick<LogoProps, 'size' | 'theme'>) {
  return (
    <Logo
      variant="stacked"
      size={size}
      theme={theme}
      showText={true}
      textPosition="below"
    />
  );
}

export function LoadingLogo({ size = 'lg' }: Pick<LogoProps, 'size'>) {
  return (
    <div className="flex flex-col items-center justify-center space-y-4">
      <Logo
        variant="icon"
        size={size}
        loading={false}
        className="animate-spin"
      />
      <div className="text-sm text-gray-600 animate-pulse">
        Loading A2A World...
      </div>
    </div>
  );
}

export function FaviconLogo() {
  return (
    <Logo
      variant="icon"
      size="custom"
      width={16}
      height={16}
      showText={false}
    />
  );
}

export default Logo;