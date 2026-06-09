#!/usr/bin/env bun
/**
 * Build unfake-core WASM for the browser worker.
 * Output: browser-tool/vendor/unfake-core/
 *
 * Requires: rust wasm32-unknown-unknown target, wasm-bindgen-cli (same version as crate).
 */
import { mkdir } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const outDir = join(root, 'browser-tool', 'vendor', 'unfake-core');
const crate = join(root, 'crates', 'unfake-wasm');

function run(cmd, args, opts = {}) {
  const result = spawnSync(cmd, args, {
    stdio: 'inherit',
    cwd: root,
    shell: process.platform === 'win32',
    ...opts,
  });
  if (result.status !== 0) {
    throw new Error(`${cmd} ${args.join(' ')} failed (${result.status})`);
  }
}

await mkdir(outDir, { recursive: true });

run('cargo', ['build', '-p', 'unfake-wasm', '--target', 'wasm32-unknown-unknown', '--release']);

const meta = spawnSync('cargo', ['metadata', '--format-version', '1', '--no-deps'], {
  encoding: 'utf8',
  cwd: root,
  shell: process.platform === 'win32',
});
const targetDir = JSON.parse(meta.stdout).target_directory;
const wasmBin = join(targetDir, 'wasm32-unknown-unknown', 'release', 'unfake_wasm.wasm');

const bindgen = process.platform === 'win32' ? 'wasm-bindgen.exe' : 'wasm-bindgen';

// Guard: wasm-bindgen CLI must match the crate's wasm-bindgen version, or the
// generated JS glue silently mismatches the .wasm ABI.
const fullMeta = spawnSync('cargo', ['metadata', '--format-version', '1'], {
  encoding: 'utf8',
  cwd: root,
  shell: process.platform === 'win32',
});
const wbVer = JSON.parse(fullMeta.stdout).packages.find((p) => p.name === 'wasm-bindgen')?.version;
const cliOut = spawnSync(bindgen, ['--version'], {
  encoding: 'utf8',
  cwd: root,
  shell: process.platform === 'win32',
});
const cliVer = (cliOut.stdout || '').trim().replace(/^wasm-bindgen\s+/, '');
if (wbVer && cliVer && wbVer !== cliVer) {
  throw new Error(
    `wasm-bindgen CLI ${cliVer} != crate wasm-bindgen ${wbVer}.\n` +
    `Fix: cargo install -f wasm-bindgen-cli --version ${wbVer}`,
  );
}

run(bindgen, [wasmBin, '--out-dir', outDir, '--target', 'web', '--no-typescript']);

console.log(`WASM -> ${outDir}`);
