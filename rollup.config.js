import { nodeResolve } from '@rollup/plugin-node-resolve';
import { terser } from 'rollup-plugin-terser';
import dts from 'rollup-plugin-dts';

const config = [
  // Main library build
  {
    input: 'lib/index.js',
    output: [
      {
        file: 'dist/unfake.js',
        format: 'umd',
        name: 'UnfakeJS',
        sourcemap: true,
        globals: {
          'image-q': 'IQ',
          'potrace-wasm': 'PotraceWasm'
        }
      },
      {
        file: 'dist/unfake.mjs',
        format: 'esm',
        sourcemap: true
      }
    ],
    external: ['image-q', 'potrace-wasm'],
    plugins: [
      nodeResolve(),
      terser({
        compress: {
          drop_console: false, // Keep console logs for debugging
          drop_debugger: true
        },
        mangle: {
          reserved: ['cv', 'cvReady'] // Don't mangle OpenCV globals
        }
      })
    ]
  },
  
  // TypeScript definitions
  {
    input: 'lib/index.js',
    output: {
      file: 'dist/index.d.ts',
      format: 'es'
    },
    plugins: [dts()]
  }
];

export default config; 