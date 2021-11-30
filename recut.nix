{ nixpkgs, mcp3d, openvdb, tiff, ... }:
let
  pkgs = import nixpkgs {
    system = "x86_64-linux";
    #overlays = [ (import ./overlay.nix) ];
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

  cmakeFlags = ["-DUSE_VDB=ON -DLOG=ON -DLOG_FULL=OFF -DFULL_PRINT=OFF -DUSE_OMP_BLOCK=ON -DUSE_OMP_INTERVAL=ON -DTEST_ALL_BENCHMARKS=OFF -DCMAKE_BUILD_TYPE=Debug -DCMAKE_MODULE_PATH=${openvdb.defaultPackage.x86_64-linux}/lib/cmake/OpenVDB -DTINYTIFF_PATH=${tiff.defaultPackage.x86_64-linux}/lib/cmake/TinyTIFFShared"];

  nativeBuildInputs = [ cmake gcc11 ];

  buildInputs = [
    python38Packages.matplotlib
    python38Packages.pandas
    range-v3
    gtest
    libtiff

    # optional dependencies
    openvdb.defaultPackage.x86_64-linux
    tiff.defaultPackage.x86_64-linux
    #mcp3d.defaultPackage.x86_64-linux
    gbenchmark

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

    # propagated binaries from openvdb
    cp ${openvdb.defaultPackage.x86_64-linux}/bin/vdb_view $out/bin
    cp ${openvdb.defaultPackage.x86_64-linux}/bin/vdb_print $out/bin
    # cp ${openvdb.defaultPackage.x86_64-linux}/bin/vdb_render $out/bin
    # cp ${openvdb.defaultPackage.x86_64-linux}/bin/vdb_lod $out/bin
    # cp ${openvdb.defaultPackage.x86_64-linux}/bin/vdb_test $out/bin
  '';

}
