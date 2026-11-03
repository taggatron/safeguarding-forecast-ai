import React from 'react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const Card: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  className,
  children,
  ...props
}) => (
  <div
    className={cn(
      'rounded-2xl border border-slate-200/80 bg-white shadow-xs transition-all duration-200',
      className
    )}
    {...props}
  >
    {children}
  </div>
);

export const CardHeader: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  className,
  children,
  ...props
}) => (
  <div className={cn('flex flex-col space-y-1.5 p-5 pb-3', className)} {...props}>
    {children}
  </div>
);

export const CardTitle: React.FC<React.HTMLAttributes<HTMLHeadingElement>> = ({
  className,
  children,
  ...props
}) => (
  <h3
    className={cn(
      'font-display text-base font-bold tracking-tight text-slate-900',
      className
    )}
    {...props}
  >
    {children}
  </h3>
);

export const CardDescription: React.FC<React.HTMLAttributes<HTMLParagraphElement>> = ({
  className,
  children,
  ...props
}) => (
  <p className={cn('text-xs text-slate-500 leading-relaxed', className)} {...props}>
    {children}
  </p>
);

export const CardContent: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  className,
  children,
  ...props
}) => (
  <div className={cn('p-5 pt-2', className)} {...props}>
    {children}
  </div>
);

export const Badge: React.FC<
  React.HTMLAttributes<HTMLSpanElement> & {
    variant?: 'default' | 'teal' | 'blue' | 'purple' | 'amber' | 'emerald' | 'rose' | 'slate';
  }
> = ({ className, variant = 'default', children, ...props }) => {
  const variants: Record<string, string> = {
    default: 'bg-slate-900 text-white',
    teal: 'bg-teal-50 text-teal-800 border border-teal-200',
    blue: 'bg-blue-50 text-blue-800 border border-blue-200',
    purple: 'bg-purple-50 text-purple-800 border border-purple-200',
    amber: 'bg-amber-50 text-amber-900 border border-amber-300',
    emerald: 'bg-emerald-50 text-emerald-800 border border-emerald-200',
    rose: 'bg-rose-50 text-rose-800 border border-rose-200',
    slate: 'bg-slate-100 text-slate-700 border border-slate-200',
  };
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-semibold',
        variants[variant],
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
};

export const Button: React.FC<
  React.ButtonHTMLAttributes<HTMLButtonElement> & {
    variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger';
    size?: 'sm' | 'md' | 'lg';
  }
> = ({ className, variant = 'primary', size = 'md', children, ...props }) => {
  const variants: Record<string, string> = {
    primary:
      'bg-slate-900 text-white hover:bg-slate-800 active:bg-slate-950 shadow-xs disabled:opacity-50',
    secondary:
      'bg-teal-600 text-white hover:bg-teal-700 active:bg-teal-800 shadow-xs disabled:opacity-50',
    outline:
      'border border-slate-300 bg-white text-slate-700 hover:bg-slate-50 active:bg-slate-100 disabled:opacity-50',
    ghost: 'text-slate-600 hover:bg-slate-100 hover:text-slate-900 disabled:opacity-50',
    danger: 'bg-rose-600 text-white hover:bg-rose-700 disabled:opacity-50',
  };
  const sizes: Record<string, string> = {
    sm: 'px-2.5 py-1.5 text-xs rounded-lg',
    md: 'px-3.5 py-2 text-xs sm:text-sm rounded-xl font-medium',
    lg: 'px-5 py-2.5 text-sm rounded-xl font-semibold',
  };
  return (
    <button
      className={cn(
        'inline-flex items-center justify-center gap-2 transition-colors cursor-pointer disabled:cursor-not-allowed',
        variants[variant],
        sizes[size],
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
};

