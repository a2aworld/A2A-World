/**
 * A2A World Platform - Header Component
 * 
 * Enhanced header component with A2A Star Logo branding and navigation.
 */

import React from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';
import { clsx } from 'clsx';
import { Menu, X, Settings, User } from 'lucide-react';
import { HeaderLogo } from '../branding/Logo';

export interface HeaderProps {
  /**
   * Whether to show mobile menu toggle
   */
  showMobileMenu?: boolean;
  /**
   * Whether to show user menu
   */
  showUserMenu?: boolean;
  /**
   * Additional CSS classes
   */
  className?: string;
  /**
   * Current user (if authenticated)
   */
  user?: {
    name: string;
    email: string;
    avatar?: string;
  };
  /**
   * Mobile menu state
   */
  mobileMenuOpen?: boolean;
  /**
   * Mobile menu toggle handler
   */
  onMobileMenuToggle?: () => void;
}

const navigationLinks = [
  { href: '/', label: 'Dashboard', description: 'System overview and metrics' },
  { href: '/maps', label: 'Maps', description: 'Interactive geospatial visualization' },
  { href: '/patterns', label: 'Patterns', description: 'Discovered pattern analysis' },
  { href: '/agents', label: 'Agents', description: 'AI agent management' },
  { href: '/data', label: 'Data', description: 'Dataset upload and management' }
];

export function Header({
  showMobileMenu = true,
  showUserMenu = false,
  className,
  user,
  mobileMenuOpen = false,
  onMobileMenuToggle
}: HeaderProps) {
  const router = useRouter();

  const isActivePath = (href: string) => {
    if (href === '/') {
      return router.pathname === '/';
    }
    return router.pathname.startsWith(href);
  };

  return (
    <header className={clsx(
      'bg-white shadow-sm border-b border-gray-200 sticky top-0 z-50',
      className
    )}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo and Brand */}
          <div className="flex items-center">
            <HeaderLogo size="md" clickable={true} />
          </div>
          
          {/* Desktop Navigation */}
          <nav className="hidden lg:flex space-x-8">
            {navigationLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={clsx(
                  'inline-flex items-center px-1 pt-1 text-sm font-medium border-b-2 transition-colors',
                  isActivePath(link.href)
                    ? 'border-blue-500 text-gray-900'
                    : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                )}
                aria-label={link.description}
              >
                {link.label}
              </Link>
            ))}
          </nav>

          {/* Right side actions */}
          <div className="flex items-center space-x-4">
            {/* User Menu */}
            {showUserMenu && user && (
              <div className="relative">
                <button className="flex items-center text-sm rounded-full focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">
                  <span className="sr-only">Open user menu</span>
                  {user.avatar ? (
                    <img
                      className="h-8 w-8 rounded-full"
                      src={user.avatar}
                      alt={user.name}
                    />
                  ) : (
                    <div className="h-8 w-8 rounded-full bg-gray-300 flex items-center justify-center">
                      <User className="h-4 w-4 text-gray-600" />
                    </div>
                  )}
                </button>
              </div>
            )}

            {/* Settings Button */}
            <button className="text-gray-400 hover:text-gray-600 p-2 rounded-md hover:bg-gray-100 transition-colors">
              <Settings className="h-5 w-5" />
              <span className="sr-only">Settings</span>
            </button>

            {/* Mobile menu button */}
            {showMobileMenu && (
              <button
                onClick={onMobileMenuToggle}
                className="lg:hidden text-gray-400 hover:text-gray-600 p-2 rounded-md hover:bg-gray-100 transition-colors"
                aria-expanded={mobileMenuOpen}
              >
                <span className="sr-only">Open main menu</span>
                {mobileMenuOpen ? (
                  <X className="h-6 w-6" />
                ) : (
                  <Menu className="h-6 w-6" />
                )}
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Mobile Navigation Menu */}
      {showMobileMenu && mobileMenuOpen && (
        <div className="lg:hidden">
          <div className="pt-2 pb-3 space-y-1 bg-white border-t border-gray-200">
            {navigationLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={clsx(
                  'block px-4 py-2 text-base font-medium border-l-4 transition-colors',
                  isActivePath(link.href)
                    ? 'bg-blue-50 border-blue-500 text-blue-700'
                    : 'border-transparent text-gray-600 hover:bg-gray-50 hover:border-gray-300 hover:text-gray-800'
                )}
                onClick={onMobileMenuToggle}
              >
                <div>
                  <div>{link.label}</div>
                  <div className="text-sm text-gray-500 mt-1">{link.description}</div>
                </div>
              </Link>
            ))}
          </div>
          
          {/* Mobile User Menu */}
          {showUserMenu && user && (
            <div className="pt-4 pb-3 border-t border-gray-200">
              <div className="px-4">
                <div className="text-base font-medium text-gray-800">{user.name}</div>
                <div className="text-sm text-gray-500">{user.email}</div>
              </div>
            </div>
          )}
        </div>
      )}
    </header>
  );
}

export default Header;