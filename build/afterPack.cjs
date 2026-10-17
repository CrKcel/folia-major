const fs = require('node:fs/promises');
const path = require('node:path');

// build/afterPack.cjs

const ONNX_BIN_RELATIVE_PATH = path.join(
  'app.asar.unpacked',
  'node_modules',
  'onnxruntime-node',
  'bin',
  'napi-v6',
);

// node-llama-cpp 是带 top-level await 的 ESM 包：Electron 的 asar 补丁只覆盖 CJS require，
// ESM 裸导入在 asar 内/外都解析不到正常布局里的依赖（它的 28 个运行时依赖在 asar 里，它自己
// 被 unpack 出去，两边的解析都会断）。所以打包时不依赖 asar 的 smart-unpack，而是在 afterPack
// 把 node-llama-cpp 的完整依赖闭包 + 目标平台的 @node-llama-cpp 二进制包复制成一份真实的
// resources/llm-runtime/node_modules 目录，worker 用绝对路径 import 它。
const LLM_RUNTIME_RELATIVE_PATH = path.join('resources', 'llm-runtime', 'node_modules');

// electron 平台名到 @node-llama-cpp 包名前缀的映射。win32 的包叫 win-*。
const NODE_LLAMA_CPP_PLATFORM_PREFIXES = {
  darwin: 'darwin',
  linux: 'linux',
  win32: 'win',
};

// electron-builder 的 Arch 枚举，按序号展开。刻意不 require('builder-util')：那只是
// electron-builder 的传递依赖，能解析到纯属 npm 扁平提升的副作用，依赖树一变就会在打包
// 末期才炸。这五个值是 electron-builder 的公开契约，比提升可靠。
const ARCH_NAMES = ['ia32', 'x64', 'armv7l', 'arm64', 'universal'];

async function pathExists(candidate) {
  try {
    await fs.access(candidate);
    return true;
  } catch {
    return false;
  }
}

// Keep only the native ONNX Runtime binaries usable by the package being built.
async function pruneOnnxRuntimeBinaries(context) {
  const resourcesDir = context.packager.getResourcesDir(context.appOutDir);
  const binariesDir = path.join(resourcesDir, ONNX_BIN_RELATIVE_PATH);
  if (!(await pathExists(binariesDir))) return;

  const targetPlatform = context.electronPlatformName;
  const targetArch = ARCH_NAMES[context.arch];
  if (!targetArch) {
    // 不认识的架构就整包留着。少几百 MB 好过删光目标架构自己的 .so。
    console.warn(`[afterPack] unknown Arch ordinal ${context.arch}, keeping every onnxruntime binary`);
    return;
  }
  const keptArchitectures = targetArch === 'universal'
    ? new Set(['x64', 'arm64'])
    : new Set([targetArch]);

  for (const platformEntry of await fs.readdir(binariesDir, { withFileTypes: true })) {
    if (!platformEntry.isDirectory()) continue;
    const platformDir = path.join(binariesDir, platformEntry.name);

    if (platformEntry.name !== targetPlatform) {
      await fs.rm(platformDir, { recursive: true, force: true });
      continue;
    }

    for (const archEntry of await fs.readdir(platformDir, { withFileTypes: true })) {
      if (archEntry.isDirectory() && !keptArchitectures.has(archEntry.name)) {
        await fs.rm(path.join(platformDir, archEntry.name), { recursive: true, force: true });
      }
    }
  }

  await warnIfTargetHasNoBinary(binariesDir, targetPlatform, keptArchitectures);
}

/**
 * 目标平台/架构在 onnxruntime-node 里根本没有原生库时，在构建日志里喊一声。
 *
 * 现在就有一例：onnxruntime-node 1.29 的 darwin 只发 arm64，而 build.mac.target 仍然构建 x64，
 * 于是 mac x64 包裁剪后这里是空的。功能上不是回归——原先塞进去的是 arm64 的 .node，Intel
 * Electron 一样 require 不动，worker.cjs 的顶层 catch 会把它降级成渲染层估算器——但沉默地发一个
 * 分析功能失效的包不该靠人去发现。上游哪天再砍掉一个架构，这行会在打包时就说出来。
 */
async function warnIfTargetHasNoBinary(binariesDir, targetPlatform, keptArchitectures) {
  const platformDir = path.join(binariesDir, targetPlatform);
  const present = (await pathExists(platformDir))
    ? (await fs.readdir(platformDir, { withFileTypes: true }))
      .filter(entry => entry.isDirectory())
      .map(entry => entry.name)
    : [];

  const missing = [...keptArchitectures].filter(arch => !present.includes(arch));
  if (missing.length === 0) return;

  console.warn(
    `[afterPack] onnxruntime-node ships no binary for ${targetPlatform}/${missing.join(', ')}`
    + ' - beat detection will fall back to the renderer estimators in this package',
  );
}

// Build resources/llm-runtime: node-llama-cpp's full runtime dependency closure plus the
// @node-llama-cpp binary packages for the target platform, copied as real files out of the build
// machine's node_modules. The worker imports the runtime from here by absolute path (see
// electron/llm/worker.cjs and FOLIA_LLM_RUNTIME_DIR).
async function buildLlmRuntime(context) {
  const projectNodeModules = path.join(context.packager.projectDir, 'node_modules');
  if (!(await pathExists(path.join(projectNodeModules, 'node-llama-cpp')))) return;

  const destNodeModules = path.join(context.appOutDir, LLM_RUNTIME_RELATIVE_PATH);
  const copied = new Set();

  /** baseDir 本身是一个包目录；收集它内部嵌套 node_modules 里的所有包目录（含 @scope）。 */
  const collectPackageDirs = async (baseDir) => {
    const found = [baseDir];
    const nestedRoot = path.join(baseDir, 'node_modules');
    if (!(await pathExists(nestedRoot))) return found;
    for (const entry of await fs.readdir(nestedRoot, { withFileTypes: true })) {
      if (!entry.isDirectory() || entry.name === '.bin') continue;
      const nestedPkgs = entry.name.startsWith('@')
        ? (await fs.readdir(path.join(nestedRoot, entry.name), { withFileTypes: true }))
          .filter(scoped => scoped.isDirectory())
          .map(scoped => path.join(nestedRoot, entry.name, scoped.name))
        : [path.join(nestedRoot, entry.name)];
      for (const pkgDir of nestedPkgs) {
        found.push(...await collectPackageDirs(pkgDir));
      }
    }
    return found;
  };

  /**
   * Walks the closure over the packages AS LAID OUT IN THE DESTINATION. npm hoists most deps
   * top-level but nests version conflicts (e.g. stdout-update's older string-width), and a nested
   * copy's deps resolve top-level even when the top-level name was pulled in by a different major
   * version - so every package in the copied subtree gets the same treatment, not just the
   * top-level ones.
   */
  const walkDepsOf = async (destPkgDir, visited) => {
    if (visited.has(destPkgDir)) return;
    visited.add(destPkgDir);
    let pkg;
    try {
      pkg = JSON.parse(await fs.readFile(path.join(destPkgDir, 'package.json'), 'utf8'));
    } catch {
      console.warn(`[afterPack] cannot read ${destPkgDir}/package.json, skipping its deps`);
      return;
    }
    for (const dep of Object.keys(pkg.dependencies ?? {})) {
      const depInDest = path.join(destNodeModules, dep);
      if (await pathExists(depInDest)) {
        await walkDepsOf(depInDest, visited);
        continue;
      }
      const projectTopLevel = path.join(projectNodeModules, dep);
      if (await pathExists(projectTopLevel)) {
        await fs.cp(projectTopLevel, depInDest, { recursive: true });
        copied.add(dep);
        for (const dir of await collectPackageDirs(depInDest)) {
          await walkDepsOf(dir, visited);
        }
        continue;
      }
      const relToDest = path.relative(destNodeModules, destPkgDir);
      const nestedOriginal = path.join(projectNodeModules, relToDest, 'node_modules', dep);
      if (await pathExists(nestedOriginal)) {
        // Already inside the copied subtree; only its deps still need the walk.
        await walkDepsOf(path.join(destPkgDir, 'node_modules', dep), visited);
        continue;
      }
      console.warn(`[afterPack] node-llama-cpp dep ${dep} (from ${relToDest}) not found - local LLM may fail`);
    }
  };

  await fs.cp(path.join(projectNodeModules, 'node-llama-cpp'), path.join(destNodeModules, 'node-llama-cpp'), { recursive: true });
  copied.add('node-llama-cpp');
  const visited = new Set();
  for (const dir of await collectPackageDirs(path.join(destNodeModules, 'node-llama-cpp'))) {
    await walkDepsOf(dir, visited);
  }

  const prefix = NODE_LLAMA_CPP_PLATFORM_PREFIXES[context.electronPlatformName];
  const binariesRoot = path.join(projectNodeModules, '@node-llama-cpp');
  if (!prefix || !(await pathExists(binariesRoot))) {
    console.warn(`[afterPack] no @node-llama-cpp binaries for ${context.electronPlatformName} - local LLM will fail in this package`);
    return;
  }
  for (const entry of await fs.readdir(binariesRoot, { withFileTypes: true })) {
    if (!entry.isDirectory() || !entry.name.startsWith(prefix)) continue;
    const name = `@node-llama-cpp/${entry.name}`;
    await fs.cp(path.join(binariesRoot, entry.name), path.join(destNodeModules, name), { recursive: true });
    copied.add(name);
  }

  console.log(`[afterPack] llm-runtime built: ${copied.size} packages -> ${LLM_RUNTIME_RELATIVE_PATH}`);
}

exports.default = async function afterPack(context) {
  await pruneOnnxRuntimeBinaries(context);
  await buildLlmRuntime(context);
};
