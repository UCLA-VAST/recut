{ nixpkgs, mcp3d, ... }:
let
  pkgs = import nixpkgs {
    system = "x86_64-linux";
    overlays = [ (import ./overlay.nix) ];
  };
in
  with pkgs;
  stdenv.mkDerivation {
    name = "recut";
    version = "0.9.0";

  # https://nixos.org/nix/manual/#builtin-filterSource
  src = builtins.filterSource
  (path: type: lib.cleanSourceFilter path type
  && baseNameOf path != "build"
  && baseNameOf path != "bin/*"
  && baseNameOf path != "data/*") ./.;

  enableParallelBuilding = true;

  # if test all benchmarks on, you can define these for benchmarks
  TEST_IMAGE = "/curr/kdmarrett/data/tcase6_image";
  TEST_MARKER = "/curr/kdmarrett/data/tcase6_marker";

  cmakeFlags = ["-DUSE_VDB=ON -DLOG=ON -DLOG_FULL=ON -DFULL_PRINT=ON -DUSE_OMP_BLOCK=OFF -DUSE_MCP3D=ON -DTEST_ALL_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_MODULE_PATH=${openvdb}/lib/cmake/OpenVDB"];

  nativeBuildInputs = [ cmake ];

  buildInputs = [
    python38Packages.matplotlib
    range-v3
    gtest

    # optional dependencies
    mcp3d.defaultPackage.x86_64-linux
    gbenchmark

    # OpenVDB dependencies
    openvdb
    openexr
    tbb
    c-blosc

    # For debug purposes only:
    # warning leaving breakpointHook on
    # will cause github actions to hang, if there are any failures
    # always comment it out before pushing
    # breakpointHook
  ];

  # more debug info
  dontStrip = true;
  enableDebugging = true;

  doInstallCheck = true;
  # You have to run installcheck step first such that the relevant files used at runtime
  # are in the data directory before running any other tests
  installCheckPhase = ''
    echo; echo;
    mkdir $out/data;
    make installcheck;
  '';

}
