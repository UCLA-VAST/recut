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

  # used for automated testing 
  doCheck = true;
  enableParallelBuilding = true;

  cmakeFlags = ["-DFROM_NIX_BUILD=ON -DRECUT_ENABLE_EXECUTABLE=ON -DLOG=ON -DUSE_MCP3D=ON -DTEST_ALL_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Debug"];

  nativeBuildInputs = [ cmake ];

  buildInputs = [ 
    python38Packages.matplotlib 
    gtest
    gbenchmark
    range-v3
    mcp3d.defaultPackage.x86_64-linux
    # warning leaving breakpointHook on 
    # will cause github actions to hang, if there are any failures
    # always comment it out before pushing
    #breakpointHook
  ];

  # You have to run the install step first such that the relevant files used at runtime
  # are in the data directory before running any other tests
  checkPhase="./recut_test --gtest_also_run_disabled_tests --gtest_filter=Install.\"*\"; ./recut_test --gtest_output=json:../data/test_detail.json | tee ../data/test_detail.log";

  installPhase = ''
    mkdir -p $out/bin
    cp recut_test $out/bin/recut_test
    cp recut $out/bin/recut
    # test data included by default, so recut_test can be run by users 
    mkdir -p $out/data;
    cp -ra ../data/* $out/data;
    cp ../data/*.json $out/data/
    ## if storing interval_base in nix dir permanently uncomment below
    # cp ../data/*.bin $out/data/
  '';

}
