import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// File validation
export function validateFile(
  file: File,
  maxSizeMB: number = 50,
  allowedTypes: string[] = []
): { valid: boolean; error?: string } {
  // Check size
  const maxBytes = maxSizeMB * 1024 * 1024;
  if (file.size > maxBytes) {
    return {
      valid: false,
      error: `File size must be less than ${maxSizeMB}MB`,
    };
  }

  // Check type if specified
  if (allowedTypes.length > 0) {
    const fileType = file.type;
    const isAllowed = allowedTypes.some(type => fileType.includes(type));
    
    if (!isAllowed) {
      return {
        valid: false,
        error: `File type must be one of: ${allowedTypes.join(', ')}`,
      };
    }
  }

  return { valid: true };
}

// Format file size
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Copy to clipboard
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (err) {
    console.error('Failed to copy:', err);
    return false;
  }
}
