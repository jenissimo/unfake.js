#!/usr/bin/env bun
/**
 * Build static site for itch.io / releases.
 *
 * 1. Copy browser-tool/ → releases/dist/  (deployable site root)
 * 2. Optional --zip: pack releases/dist/ → .itch-upload/unfaker-{version}.zip
 *
 * Usage (repo root):
 *   bun run deploy          # copy only
 *   bun run deploy:zip      # copy + zip for itch upload
 */

import { cp, mkdir, readdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { join, relative } from 'node:path';
import { zipSync } from 'fflate';
import { $ } from 'bun';

const ROOT = join(import.meta.dir, '..');
const SRC = join(ROOT, 'browser-tool');
const DIST = join(ROOT, 'releases', 'dist');
const ITCH_DIR = join(ROOT, '.itch-upload');

interface PackageJson {
  name: string;
  version: string;
}

async function readVersion(): Promise<string> {
  const raw = await readFile(join(ROOT, 'package.json'), 'utf8');
  const pkg = JSON.parse(raw) as PackageJson;
  if (!pkg.version) throw new Error('package.json missing "version"');
  return pkg.version;
}

async function copyBrowserTool(): Promise<void> {
  await rm(DIST, { recursive: true, force: true });
  await mkdir(join(ROOT, 'releases'), { recursive: true });
  await cp(SRC, DIST, { recursive: true, force: true });
  console.log(`[deploy] copied ${relative(ROOT, SRC)} → ${relative(ROOT, DIST)}`);
}

/** Walk directory and collect relative paths for zip (posix separators). */
async function collectFiles(dir: string, base = dir): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true });
  const out: string[] = [];
  for (const entry of entries) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      out.push(...(await collectFiles(full, base)));
    } else if (entry.isFile()) {
      out.push(relative(base, full).split('\\').join('/'));
    }
  }
  return out;
}

async function buildZip(version: string): Promise<string> {
  const files = await collectFiles(DIST);
  const archive: Record<string, Uint8Array> = {};

  for (const rel of files) {
    const abs = join(DIST, rel);
    const info = await stat(abs);
    if (!info.isFile()) continue;
    const data = await readFile(abs);
    archive[rel] = new Uint8Array(data);
  }

  const zipped = zipSync(archive, { level: 6 });
  await mkdir(ITCH_DIR, { recursive: true });

  const zipName = `unfaker-${version}.zip`;
  const zipPath = join(ITCH_DIR, zipName);
  await writeFile(zipPath, zipped);

  const mb = (zipped.byteLength / (1024 * 1024)).toFixed(2);
  console.log(`[deploy] wrote ${relative(ROOT, zipPath)} (${mb} MB, ${files.length} files)`);
  return zipPath;
}

async function main(): Promise<void> {
  const zip = process.argv.includes('--zip');
  const version = await readVersion();

  console.log(`[deploy] unfake.js v${version}`);

  try {
    await stat(SRC);
  } catch {
    throw new Error(`Source not found: ${SRC}`);
  }

  console.log('[deploy] building unfake-core WASM…');
  await $`bun run core:wasm`.cwd(ROOT).quiet();

  await copyBrowserTool();

  if (zip) {
    await buildZip(version);
    console.log('[deploy] Upload .itch-upload/*.zip to itch.io (project root = zip contents)');
  } else {
    console.log('[deploy] Done. Run with --zip to create itch.io archive.');
  }
}

main().catch((err) => {
  console.error('[deploy] failed:', err instanceof Error ? err.message : err);
  process.exit(1);
});
