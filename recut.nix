{ nixpkgs, mcp3d, ... }:
let
  pkgs = import nixpkgs { system = "x86_64-linux"; };
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

  cmakeFlags = ["-DLOG=ON -DLOG_FULL=ON -DFULL_PRINT=OFF -DUSE_OMP_BLOCK=ON -DUSE_MCP3D=ON -DTEST_ALL_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Debug"];

  nativeBuildInputs = [ cmake ];

  buildInputs = [ 
    python38Packages.matplotlib 
    gtest
    gbenchmark
    range-v3
    #openvdb
    mcp3d.defaultPackage.x86_64-linux
    # warning leaving breakpointHook on 
    # will cause github actions to hang, if there are any failures
    # always comment it out before pushing
    #breakpointHook
  ];

  doInstallCheck = true;
  # You have to run installcheck step first such that the relevant files used at runtime
  # are in the data directory before running any other tests
  installCheckPhase = ''
    mkdir $out/data;
    make installcheck;
    ./recut_test;
    ./recut_test --gtest_also_run_disabled_tests --gtest_filter='*'.'*ChecksIf*/63';
    '';

}
