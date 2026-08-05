/**
 * WebGL 2.0 Hardware Acceleration Manager for NVIDIA dedicated GPUs (dGPU / GTX 950M).
 * Queries UNMASKED_RENDERER_WEBGL and formats a clean user-friendly GPU name.
 */
export interface GPUInfo {
  vendor: string;
  renderer: string;
  cleanName: string;
  isDedicatedNvidia: boolean;
  webglVersion: string;
}

export function detectGPUInfo(): GPUInfo {
  try {
    const canvas = document.createElement('canvas');
    const gl =
      (canvas.getContext('webgl2', { powerPreference: 'high-performance' }) as WebGL2RenderingContext) ||
      (canvas.getContext('webgl', { powerPreference: 'high-performance' }) as WebGLRenderingContext);

    if (!gl) {
      return {
        vendor: 'Canvas 2D Fallback',
        renderer: 'Software / Basic GPU',
        cleanName: 'Rendu Standard',
        isDedicatedNvidia: false,
        webglVersion: 'None',
      };
    }

    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
    let vendor = gl.getParameter(gl.VENDOR);
    let renderer = gl.getParameter(gl.RENDERER);

    if (debugInfo) {
      vendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) || vendor;
      renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) || renderer;
    }

    const rawRenderer = String(renderer);
    const isDedicatedNvidia =
      rawRenderer.toUpperCase().includes('NVIDIA') ||
      rawRenderer.toUpperCase().includes('GEFORCE') ||
      rawRenderer.toUpperCase().includes('GTX') ||
      rawRenderer.toUpperCase().includes('RTX');

    // Clean raw renderer strings (e.g. remove "ANGLE (NVIDIA, Direct3D11..." prefix)
    let cleanName = rawRenderer;
    if (isDedicatedNvidia) {
      const match = rawRenderer.match(/(NVIDIA\s+GeForce\s+[A-Za-z0-9\s]+|NVIDIA\s+[A-Za-z0-9\s]+|GTX\s+[0-9]+M?|RTX\s+[0-9]+M?)/i);
      if (match) {
        cleanName = match[0].trim();
      } else {
        cleanName = 'NVIDIA GPU Dedicated';
      }
    } else {
      cleanName = rawRenderer.replace(/^ANGLE\s*\([^)]*\)\s*/i, '').trim() || 'Accélération GPU';
    }

    return {
      vendor: String(vendor),
      renderer: rawRenderer,
      cleanName,
      isDedicatedNvidia,
      webglVersion: gl instanceof WebGL2RenderingContext ? 'WebGL 2.0' : 'WebGL 1.0',
    };
  } catch (e) {
    return {
      vendor: 'Inconnu',
      renderer: 'Mode standard',
      cleanName: 'Rendu Graphique Standard',
      isDedicatedNvidia: false,
      webglVersion: 'Indisponible',
    };
  }
}
