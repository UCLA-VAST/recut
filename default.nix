with import <nixpkgs> {};
let
  unstable = import (fetchTarball https://nixos.org/channels/nixos-unstable/nixexprs.tar.xz) { };
  in

stdenv.mkDerivation {
  name = "recut";

  # https://nixos.org/nix/manual/#builtin-filterSource
  src = builtins.filterSource 
    (path: type: lib.cleanSourceFilter path type 
    && baseNameOf path != "build"
    && baseNameOf path != "bin"
    && baseNameOf path != "data") ./.;

  nativeBuildInputs = [ cmake ];
  buildInputs = [ openssl boost libtiff unstable.mpich 
    python38Packages.matplotlib clang_10 llvmPackages.openmp ];

  buildPhase = "mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DUSE_MCP3D=ON && echo; echo; pwd; make -j 56 && make install -j 56";

  checkPhase="echo; pwd; cd ../bin && ./recut_test --gtest_output=json:../data/test_detail.json ../data/test_detail.log";

  installPhase = ''
    mkdir -p $out/bin
    cp bin/* $out/bin/
  '';
}
