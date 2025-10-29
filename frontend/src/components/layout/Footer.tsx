/**
 * A2A World Platform - Footer Component
 * 
 * Comprehensive footer component with branding, contact information,
 * and professional styling for all pages.
 */

import React from 'react';
import Link from 'next/link';
import { clsx } from 'clsx';
import { Mail, Heart, ExternalLink } from 'lucide-react';
import { FooterLogo } from '../branding/Logo';

export interface FooterProps {
  /**
   * Footer variant
   */
  variant?: 'default' | 'minimal' | 'expanded';
  /**
   * Footer theme
   */
  theme?: 'light' | 'dark';
  /**
   * Additional CSS classes
   */
  className?: string;
  /**
   * Whether to show social media links
   */
  showSocial?: boolean;
  /**
   * Whether to show navigation links
   */
  showNavigation?: boolean;
}

const navigationLinks = [
  { href: '/', label: 'Dashboard' },
  { href: '/maps', label: 'Maps' },
  { href: '/patterns', label: 'Patterns' },
  { href: '/agents', label: 'Agents' },
  { href: '/data', label: 'Data' }
];

const legalLinks = [
  { href: '/privacy', label: 'Privacy Policy' },
  { href: '/terms', label: 'Terms of Service' },
  { href: '/about', label: 'About' },
  { href: '/contact', label: 'Contact' }
];

const socialLinks = [
  { href: 'https://github.com/a2a-world', label: 'GitHub', icon: ExternalLink },
  { href: 'https://twitter.com/a2aworld', label: 'Twitter', icon: ExternalLink },
  { href: 'https://linkedin.com/company/a2a-world', label: 'LinkedIn', icon: ExternalLink }
];

export function Footer({
  variant = 'default',
  theme = 'dark',
  className,
  showSocial = true,
  showNavigation = true
}: FooterProps) {
  const currentYear = new Date().getFullYear();
  
  const isDark = theme === 'dark';
  const containerClasses = clsx(
    'w-full border-t',
    isDark ? 'bg-gray-900 border-gray-800 text-gray-300' : 'bg-white border-gray-200 text-gray-600',
    className
  );

  if (variant === 'minimal') {
    return (
      <footer className={containerClasses}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col sm:flex-row justify-between items-center space-y-4 sm:space-y-0">
            <div className="flex items-center space-x-4">
              <FooterLogo size="sm" theme={theme} />
            </div>
            
            <div className="flex items-center space-x-4 text-sm">
              <span>Built with <Heart className="h-3 w-3 inline text-red-500" /> using ROO CODE powered by Claude</span>
              <span>© {currentYear} A2A World. All rights reserved.</span>
            </div>
          </div>
        </div>
      </footer>
    );
  }

  return (
    <footer className={containerClasses}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {variant === 'expanded' && (
          <div className="py-12">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
              {/* Company Info */}
              <div className="md:col-span-2">
                <FooterLogo size="md" theme={theme} />
                <p className={clsx(
                  'mt-4 text-sm leading-6 max-w-md',
                  isDark ? 'text-gray-400' : 'text-gray-600'
                )}>
                  A2A World is an advanced pattern discovery platform that analyzes cultural and geospatial data 
                  to uncover hidden connections and insights across ancient sites and sacred locations.
                </p>
                <div className="mt-6">
                  <a
                    href="mailto:support@a2aworld.ai"
                    className={clsx(
                      'inline-flex items-center text-sm font-medium hover:underline',
                      isDark ? 'text-blue-400 hover:text-blue-300' : 'text-blue-600 hover:text-blue-500'
                    )}
                  >
                    <Mail className="h-4 w-4 mr-2" />
                    support@a2aworld.ai
                  </a>
                </div>
              </div>

              {/* Navigation Links */}
              {showNavigation && (
                <div>
                  <h3 className={clsx(
                    'text-sm font-semibold uppercase tracking-wider',
                    isDark ? 'text-gray-300' : 'text-gray-900'
                  )}>
                    Platform
                  </h3>
                  <ul className="mt-4 space-y-3">
                    {navigationLinks.map((link) => (
                      <li key={link.href}>
                        <Link
                          href={link.href}
                          className={clsx(
                            'text-sm hover:underline',
                            isDark ? 'text-gray-400 hover:text-gray-300' : 'text-gray-600 hover:text-gray-500'
                          )}
                        >
                          {link.label}
                        </Link>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Legal Links */}
              <div>
                <h3 className={clsx(
                  'text-sm font-semibold uppercase tracking-wider',
                  isDark ? 'text-gray-300' : 'text-gray-900'
                )}>
                  Legal
                </h3>
                <ul className="mt-4 space-y-3">
                  {legalLinks.map((link) => (
                    <li key={link.href}>
                      <Link
                        href={link.href}
                        className={clsx(
                          'text-sm hover:underline',
                          isDark ? 'text-gray-400 hover:text-gray-300' : 'text-gray-600 hover:text-gray-500'
                        )}
                      >
                        {link.label}
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* Bottom Section */}
        <div className={clsx(
          'py-6 border-t',
          isDark ? 'border-gray-800' : 'border-gray-200',
          variant === 'expanded' && 'mt-0'
        )}>
          <div className="flex flex-col lg:flex-row justify-between items-center space-y-4 lg:space-y-0">
            {/* Left side - Attribution and Copyright */}
            <div className="flex flex-col sm:flex-row items-center space-y-2 sm:space-y-0 sm:space-x-6 text-sm">
              <div className="flex items-center">
                <span>Built with </span>
                <Heart className="h-3 w-3 mx-1 text-red-500" />
                <span> using </span>
                <span className={clsx(
                  'font-medium',
                  isDark ? 'text-blue-400' : 'text-blue-600'
                )}>
                  ROO CODE
                </span>
                <span> powered by </span>
                <span className={clsx(
                  'font-medium',
                  isDark ? 'text-purple-400' : 'text-purple-600'
                )}>
                  Claude
                </span>
              </div>
              
              <div className="flex items-center space-x-1">
                <span>©</span>
                <span>{currentYear}</span>
                <span className={clsx(
                  'font-medium',
                  isDark ? 'text-white' : 'text-gray-900'
                )}>
                  A2A World
                </span>
                <span>. All rights reserved.</span>
              </div>
            </div>

            {/* Right side - Contact and Social */}
            <div className="flex items-center space-x-6">
              <a
                href="mailto:support@a2aworld.ai"
                className={clsx(
                  'inline-flex items-center text-sm font-medium hover:underline',
                  isDark ? 'text-gray-400 hover:text-gray-300' : 'text-gray-600 hover:text-gray-500'
                )}
              >
                <Mail className="h-4 w-4 mr-1" />
                Support
              </a>

              {showSocial && (
                <div className="flex items-center space-x-4">
                  {socialLinks.map((social) => {
                    const Icon = social.icon;
                    return (
                      <a
                        key={social.href}
                        href={social.href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className={clsx(
                          'text-sm hover:underline',
                          isDark ? 'text-gray-400 hover:text-gray-300' : 'text-gray-600 hover:text-gray-500'
                        )}
                        aria-label={social.label}
                      >
                        <Icon className="h-4 w-4" />
                      </a>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}

export default Footer;