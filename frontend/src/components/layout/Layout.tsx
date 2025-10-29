/**
 * A2A World Platform - Layout Component
 * 
 * Main layout component that wraps all pages with branded Header and Footer.
 */

import React, { useState } from 'react';
import Head from 'next/head';
import { clsx } from 'clsx';
import Header from './Header';
import Footer from './Footer';

export interface LayoutProps {
  /**
   * Page content
   */
  children: React.ReactNode;
  /**
   * Page title (will be prefixed with "A2A World - ")
   */
  title?: string;
  /**
   * Page description for meta tag
   */
  description?: string;
  /**
   * Additional CSS classes for main content area
   */
  className?: string;
  /**
   * Footer variant
   */
  footerVariant?: 'default' | 'minimal' | 'expanded';
  /**
   * Whether to show mobile menu in header
   */
  showMobileMenu?: boolean;
  /**
   * Whether to show user menu in header
   */
  showUserMenu?: boolean;
  /**
   * Current user data
   */
  user?: {
    name: string;
    email: string;
    avatar?: string;
  };
  /**
   * Whether content should have full height
   */
  fullHeight?: boolean;
}

export function Layout({
  children,
  title = 'Pattern Discovery Platform',
  description = 'A2A World - Advanced pattern discovery platform for cultural and geospatial data analysis',
  className,
  footerVariant = 'default',
  showMobileMenu = true,
  showUserMenu = false,
  user,
  fullHeight = false
}: LayoutProps) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const handleMobileMenuToggle = () => {
    setMobileMenuOpen(!mobileMenuOpen);
  };

  const pageTitle = title === 'Pattern Discovery Platform' 
    ? 'A2A World' 
    : `${title} - A2A World`;

  return (
    <>
      <Head>
        <title>{pageTitle}</title>
        <meta name="description" content={description} />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/logo.png" />
        <meta property="og:title" content={pageTitle} />
        <meta property="og:description" content={description} />
        <meta property="og:type" content="website" />
        <meta property="og:image" content="/logo.png" />
        <meta name="twitter:card" content="summary" />
        <meta name="twitter:title" content={pageTitle} />
        <meta name="twitter:description" content={description} />
        <meta name="twitter:image" content="/logo.png" />
      </Head>

      <div className={clsx(
        'min-h-screen bg-gray-50 flex flex-col',
        fullHeight && 'h-screen'
      )}>
        <Header
          showMobileMenu={showMobileMenu}
          showUserMenu={showUserMenu}
          user={user}
          mobileMenuOpen={mobileMenuOpen}
          onMobileMenuToggle={handleMobileMenuToggle}
        />
        
        <main className={clsx(
          'flex-1',
          fullHeight && 'overflow-hidden',
          className
        )}>
          {children}
        </main>

        <Footer
          variant={footerVariant}
          theme="light"
          showSocial={true}
          showNavigation={footerVariant === 'expanded'}
        />
      </div>
    </>
  );
}

export default Layout;