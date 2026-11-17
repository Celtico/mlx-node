// Runs on `npm install`: puts in build/Release/ what the engine needs, without compiling anything and without
// Python:
//   1. mlx_node.node, the prebuilt addon for this version, from the GitHub release (built by CI);
//   2. libmlx.dylib (with the libraries it loads, such as libjaccl.dylib) and mlx.metallib, from the official mlx-metal wheel on PyPI, at the exact MLX version the
//      addon was built against (package.json «mlx»), for the newest macOS this Mac supports.
// Already there (a local build: npm run build) → nothing to do. Not an Apple Silicon Mac, or offline → a note
// and a normal exit, so installing the package never fails because of it.
const { execFileSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const root = path.join(__dirname, '..');
const pkg = require(path.join(root, 'package.json'));
const out = path.join(root, 'build', 'Release');
const needed = ['mlx_node.node', 'libmlx.dylib', 'libjaccl.dylib', 'mlx.metallib'];
const have = (name) => fs.existsSync(path.join(out, name));

/** The macOS major version from the Darwin one (Darwin 23 → 14, 24 → 15, 25 → 26). */
const macosVersion = () => {
  const darwin = Number(os.release().split('.')[0]);
  return darwin >= 25 ? darwin + 1 : darwin - 9;
};

async function download(url, file) {
  const response = await fetch(url, { redirect: 'follow' });
  if (!response.ok) throw new Error(`${response.status} ${url}`);
  fs.writeFileSync(file, Buffer.from(await response.arrayBuffer()));
}

/** The mlx-metal wheel for this MLX version that suits this Mac: the newest macOS it is built for that is not newer than this one. */
async function mlxWheel(version) {
  const response = await fetch(`https://pypi.org/pypi/mlx-metal/${version}/json`);
  if (!response.ok) throw new Error(`mlx-metal ${version} is not on PyPI (${response.status})`);
  const mac = macosVersion();
  const wheels = (await response.json()).urls
    .map((item) => ({ url: item.url, mac: Number((/macosx_(\d+)_\d+_arm64\.whl$/.exec(item.filename) || [])[1]) }))
    .filter((item) => item.mac && item.mac <= mac)
    .sort((a, b) => b.mac - a.mac);
  if (!wheels.length) throw new Error(`no mlx-metal ${version} wheel for macOS ${mac}`);
  return wheels[0].url;
}

async function main() {
  if (needed.every(have)) return;
  if (process.platform !== 'darwin' || process.arch !== 'arm64') {
    console.warn('@celtico/mlx-node: for now it runs on Apple Silicon Macs only (Linux and Windows are planned).');
    return;
  }
  fs.mkdirSync(out, { recursive: true });
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'mlx-node-'));
  try {
    if (!have('mlx_node.node')) {
      const archive = path.join(temp, 'addon.tar.gz');
      await download(`https://github.com/Celtico/mlx-node/releases/download/v${pkg.version}/mlx_node-v${pkg.version}-darwin-arm64.tar.gz`, archive);
      execFileSync('tar', ['-xzf', archive, '-C', out]);
    }
    if (!['libmlx.dylib', 'libjaccl.dylib', 'mlx.metallib'].every(have)) {
      const wheel = path.join(temp, 'mlx-metal.whl');
      await download(await mlxWheel(pkg.mlx), wheel);
      execFileSync('unzip', ['-o', '-j', '-q', wheel, 'mlx/lib/*.dylib', 'mlx/lib/mlx.metallib', '-d', out]);
    }
    console.log(`@celtico/mlx-node: ready (MLX ${pkg.mlx}).`);
  } catch (error) {
    console.warn(`@celtico/mlx-node: could not set up the engine (${error.message}).`);
    console.warn(`Build it here instead: pip install mlx==${pkg.mlx}, then npm run build (needs Xcode and cmake).`);
  } finally {
    fs.rmSync(temp, { recursive: true, force: true });
  }
}

main();
